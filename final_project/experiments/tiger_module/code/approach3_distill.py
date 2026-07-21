#!/usr/bin/env python3
"""Approach 3 — knowledge distillation from a larger teacher (ZERO manual tiger labels).

Idea (parallels TigerNet's teacher->student pseudo-labelling):
  * Teacher = a large ImageNet-pretrained ConvNeXt. ImageNet already contains a
    "tiger" class, so the teacher can score tiger-likeness with NO tiger labels
    from us. It is accurate but far too heavy for a Raspberry Pi.
  * The teacher pseudo-labels an UNLABELLED pool of frames (ATRW pool + CCT background).
  * Student = our deployed frozen MobileNetV2 features + a tiny trainable head, trained
    to imitate the teacher's soft tiger-score (distillation). The student is the small
    edge-deployable model.

We report BOTH the teacher's own zero-shot separability and the distilled student's,
evaluated on the same held-out test with TRUE labels.
"""
from __future__ import annotations
import os, sys, json, re, time, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT="/home/deploy/efficientml/efficient-ml-set/final_project"
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT,"src"))
import torchvision as tv
from torchvision.transforms import functional as TF
from PIL import Image
EMB="/home/deploy/efficientml/atrw/emb"
OUT="/home/deploy/efficientml/atrw/results_a3"; os.makedirs(OUT, exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"; SEED=0
CCT_IMG=os.path.join(ROOT,"data/raw/extracted/eccv_18_all_images_sm")

def f2(p,r):
    b2=4.0;d=b2*p+r; return 0.0 if d==0 else (1+b2)*p*r/d
def m_at(sp,sn,t):
    tp=(sp>t).sum();fn=(sp<=t).sum();fp=(sn>t).sum();tn=(sn<=t).sum()
    rec=tp/max(tp+fn,1);prec=tp/max(tp+fp,1);fpr=fp/max(fp+tn,1)
    return dict(precision=float(prec),recall=float(rec),fpr=float(fpr),f2=float(f2(prec,rec)))
def calib_f2(sp,sn):
    c=np.unique(np.concatenate([sp,sn]));bt,bb=c[0],-1
    for t in c:
        m=m_at(sp,sn,t)
        if m["f2"]>bb:bb=m["f2"];bt=float(t)
    return bt
def calib_ff(sn,b=.05): return float(np.quantile(sn,1-b))
def halves(lab,seed):
    r=np.random.default_rng(seed);a=[];b=[]
    for L in set(lab):
        idx=np.where(lab==L)[0];r.shuffle(idx);h=len(idx)//2;a+=idx[:h].tolist();b+=idx[h:].tolist()
    return np.array(sorted(a)),np.array(sorted(b))

# ---------------- teacher ----------------
def load_teacher():
    w=tv.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    m=tv.models.convnext_tiny(weights=w).eval().to(DEV)
    cats=w.meta["categories"]
    tiger_idx=[i for i,c in enumerate(cats) if re.fullmatch(r"tiger", c.strip())]
    # big-cat context helps normalise; but tiger-score = prob mass on 'tiger'
    print("[teacher] convnext_tiny; tiger classes:", [(i,cats[i]) for i in tiger_idx])
    return m, w, tiger_idx

@torch.inference_mode()
def teacher_scores(model, weights, tiger_idx, paths, bs=64, log=""):
    tfm=weights.transforms()  # ImageNet resize/centre-crop/normalise
    out=[];t0=time.time()
    for i in range(0,len(paths),bs):
        ims=[]
        for p in paths[i:i+bs]:
            im=Image.open(p).convert("RGB"); ims.append(tfm(im))
        x=torch.stack(ims).to(DEV)
        prob=F.softmax(model(x),dim=1)
        s=prob[:,tiger_idx].sum(1)
        out.append(s.float().cpu().numpy())
        if i%(bs*8)==0: print(f"  teacher {log} {i}/{len(paths)} {time.time()-t0:.0f}s",flush=True)
    return np.concatenate(out)

# ---------------- student head ----------------
class Head(nn.Module):
    def __init__(self,d=1280): super().__init__(); self.fc=nn.Linear(d,1)
    def forward(self,x): return self.fc(x).squeeze(1)

def train_student(X, soft, epochs=200, lr=1e-2, wd=1e-4):
    Xt=torch.tensor(X,dtype=torch.float32,device=DEV)
    yt=torch.tensor(soft,dtype=torch.float32,device=DEV)
    head=Head(X.shape[1]).to(DEV)
    opt=torch.optim.Adam(head.parameters(),lr=lr,weight_decay=wd)
    for e in range(epochs):
        opt.zero_grad()
        logit=head(Xt)
        loss=F.binary_cross_entropy_with_logits(logit,yt)  # soft-target distillation
        loss.backward(); opt.step()
    head.eval()
    return head

@torch.inference_mode()
def student_score(head,X):
    return head(torch.tensor(X,dtype=torch.float32,device=DEV)).cpu().numpy()

def main():
    rng=np.random.default_rng(SEED)
    # embeddings (student space) + labels
    cct=np.load(f"{EMB}/cct_cisvalclean.npz",allow_pickle=True)
    neg=cct["emb"]; neg_lab=cct["labels"].astype(str); neg_ids=cct["ids"].astype(str)
    det=np.load(f"{EMB}/atrw_detection.npz",allow_pickle=True)
    pos=det["emb"]; pos_paths=det["paths"].astype(str)
    nval,ntest=halves(neg_lab,SEED)
    neg_val,neg_test=neg[nval],neg[ntest]; neg_val_ids=neg_ids[nval]
    P=len(pos);perm=rng.permutation(P);npool=int(P*.6);nv=int(P*.2)
    pool=perm[:npool];pval=perm[npool:npool+nv];ptest=perm[npool+nv:]
    pos_val,pos_test=pos[pval],pos[ptest]

    # ---- teacher scores on the UNLABELLED distillation pool (ATRW pool + CCT val) ----
    teacher,weights,tiger_idx=load_teacher()
    atrw_pool_paths=list(pos_paths[pool])
    cct_val_paths=[os.path.join(CCT_IMG,f"{i}.jpg") for i in neg_val_ids]
    st_atrw=teacher_scores(teacher,weights,tiger_idx,atrw_pool_paths,log="atrw-pool")
    st_cct =teacher_scores(teacher,weights,tiger_idx,cct_val_paths,log="cct-val")
    # teacher zero-shot separability on TEST (for reference) -- needs teacher on test imgs
    atrw_test_paths=list(pos_paths[ptest])
    cct_test_paths=[os.path.join(CCT_IMG,f"{i}.jpg") for i in neg_ids[ntest]]
    st_pos_test=teacher_scores(teacher,weights,tiger_idx,atrw_test_paths,log="atrw-test")
    st_neg_test=teacher_scores(teacher,weights,tiger_idx,cct_test_paths,log="cct-test")
    y=np.concatenate([np.ones_like(st_pos_test),np.zeros_like(st_neg_test)])
    teacher_auc=float(roc_auc_score(y,np.concatenate([st_pos_test,st_neg_test])))
    teacher_ap =float(average_precision_score(y,np.concatenate([st_pos_test,st_neg_test])))
    print(f"[teacher] zero-shot TEST AUC={teacher_auc:.4f} AP={teacher_ap:.4f} "
          f"(pos_mean={st_pos_test.mean():.3f} neg_mean={st_neg_test.mean():.3f})")

    # ---- distill teacher soft-scores into student head (NO human tiger labels) ----
    X=np.concatenate([pos[pool],neg_val]); soft=np.concatenate([st_atrw,st_cct])
    head=train_student(X,soft)
    sp_v=student_score(head,pos_val);sn_v=student_score(head,neg_val)
    sp_t=student_score(head,pos_test);sn_t=student_score(head,neg_test)
    tf2=calib_f2(sp_v,sn_v);tff=calib_ff(sn_v,.05)
    mf2=m_at(sp_t,sn_t,tf2);mff=m_at(sp_t,sn_t,tff)
    yy=np.concatenate([np.ones_like(sp_t),np.zeros_like(sn_t)]);ss=np.concatenate([sp_t,sn_t])
    stud_auc=float(roc_auc_score(yy,ss));stud_ap=float(average_precision_score(yy,ss))
    print(f"[student-distilled] TEST AUC={stud_auc:.4f} AP={stud_ap:.4f} "
          f"F2opt={mf2['f2']:.3f} FF@5%: rec={mff['recall']:.3f} prec={mff['precision']:.3f}")

    res=dict(method="distillation_imagenet_teacher_zero_label",
             teacher="convnext_tiny(IMAGENET1K_V1)",
             n_distill_pool=int(len(X)), tiger_labels_used=0,
             teacher_zeroshot=dict(auc=teacher_auc,ap=teacher_ap,
                 pos_mean=float(st_pos_test.mean()),neg_mean=float(st_neg_test.mean())),
             student_distilled=dict(auc=stud_auc,ap=stud_ap,
                 f2opt=mf2, ff5=mff))
    json.dump(res,open(f"{OUT}/result_distill.json","w"),indent=2)
    print("APPROACH 3 DONE"); print(json.dumps(res,indent=2))

if __name__=="__main__": main()
