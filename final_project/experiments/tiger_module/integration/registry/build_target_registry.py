#!/usr/bin/env python3
"""R1 — target-registry builder. Any animal -> one head entry (w[1280], b, threshold).

Generalises T1 to an arbitrary target. Reuses the M2-space embeddings saved by T1
(m2_emb.npz: ATRW tiger `pos` + CCT `neg` with `neg_lab`). No retraining of the backbone;
adding an animal is fitting one linear head in the frozen M2 embedding space.
"""
from __future__ import annotations
import os, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

TE="/home/deploy/efficientml/tiger_embed"
d=np.load(f"{TE}/m2_emb.npz", allow_pickle=True)
cct=d["neg"]; lab=d["neg_lab"].astype(str); tiger=d["pos"]
SEED=0

def f2(p,r):
    b2=4.0; s=b2*p+r; return 0.0 if s==0 else (1+b2)*p*r/s
def metrics(sp,sn,t):
    tp=(sp>t).sum();fn=(sp<=t).sum();fp=(sn>t).sum();tn=(sn<=t).sum()
    rec=tp/max(tp+fn,1);prec=tp/max(tp+fp,1);fpr=fp/max(fp+tn,1)
    return dict(recall=float(rec),precision=float(prec),fpr=float(fpr),f2=float(f2(prec,rec)))
def halves(labels, seed):
    r=np.random.default_rng(seed); a=[]; b=[]
    for L in set(labels):
        idx=np.where(labels==L)[0]; r.shuffle(idx); h=len(idx)//2
        a+=idx[:h].tolist(); b+=idx[h:].tolist()
    return np.array(sorted(a)), np.array(sorted(b))

# stratified CCT val/test used by every native target and as tiger negatives
cval, ctest = halves(lab, SEED)

def build(name, pos_fit, neg_fit, pos_test, neg_test, source, note=""):
    X=np.concatenate([pos_fit,neg_fit]); y=np.concatenate([np.ones(len(pos_fit)),np.zeros(len(neg_fit))])
    clf=LogisticRegression(max_iter=5000,class_weight="balanced",C=1.0).fit(X,y)
    sfp=clf.decision_function(pos_fit); sfn=clf.decision_function(neg_fit)
    thr=float(np.quantile(sfn,0.95))  # 5% false-fire on the fit negatives
    spt=clf.decision_function(pos_test); snt=clf.decision_function(neg_test)
    m=metrics(spt,snt,thr)
    yy=np.concatenate([np.ones_like(spt),np.zeros_like(snt)]); ss=np.concatenate([spt,snt])
    auc=float(roc_auc_score(yy,ss))
    print(f"  {name:9s} AUC={auc:.4f} rec@5%FF={m['recall']:.3f} prec={m['precision']:.3f} "
          f"thr={thr:+.3f}  (pos_fit={len(pos_fit)} neg_fit={len(neg_fit)})")
    return dict(name=name, source=source, method="A2_linear_head_M2space", emb_l2norm=True,
                dim=int(clf.coef_.shape[1]), weight=clf.coef_[0].astype(float).tolist(),
                bias=float(clf.intercept_[0]), threshold=thr, calib="5% false-fire on fit negatives",
                metrics_test=dict(auc=auc, **m), n_pos_fit=int(len(pos_fit)), n_neg_fit=int(len(neg_fit)),
                note=note)

reg={"schema":"wildlife_trigger.target_registry.v1",
     "emb_tensor":"/Flatten_output_0","emb_space":"M2 INT8 (deployed)","l2_normalise_before_dot":True,
     "targets":[]}

print("[registry] building targets in M2 space:")
# exotic new animal — not in the 16 CCT classes
tv,tt=halves(np.zeros(len(tiger),int),SEED)  # arbitrary 50/50 on tiger
reg["targets"].append(build("tiger", tiger[tv], cct[cval], tiger[tt], cct[ctest],
                            "ATRW", "new species, absent from the 16 CCT classes"))
# native CCT animals — added by the SAME registry path, not via the 16-class head
for animal in ["bobcat","coyote","raccoon"]:
    pv=cval[lab[cval]==animal]; pt=ctest[lab[ctest]==animal]
    nv=cval[lab[cval]!=animal]; nt=ctest[lab[ctest]!=animal]
    reg["targets"].append(build(animal, cct[pv], cct[nv], cct[pt], cct[nt],
                                "CCT cis_val_clean", "native CCT class, added via registry (not the 16-class head)"))

json.dump(reg, open(f"{TE}/target_registry.json","w"), indent=2)
print(f"[registry] wrote target_registry.json with {len(reg['targets'])} targets")
