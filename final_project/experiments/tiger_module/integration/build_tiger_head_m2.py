#!/usr/bin/env python3
"""T1 — rebuild the tiger head in M2's OWN embedding space (deployment space).

The experiment used M0 FP32 embeddings. Deployment is M2 INT8, so we tap M2's pooled
1280-d embedding (`/Flatten_output_0`, the vector the classifier consumes) and rebuild
the A2 linear head there. Same split/protocol as the experiment -> fair comparison.
"""
from __future__ import annotations
import os, sys, json, time, numpy as np
import onnx, onnxruntime as ort
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

REPO="/home/deploy/efficientml/efficient-ml-set/final_project"
sys.path.insert(0, os.path.join(REPO,"src")); os.chdir(REPO)
from wildlife_trigger.data.preprocess import PreprocessConfig, normalise, preprocess_file
CFG=PreprocessConfig()

M2="/home/deploy/efficientml/tiger_embed/models/M2.onnx"
M2E="/home/deploy/efficientml/tiger_embed/models/M2_emb.onnx"
ATRW="/home/deploy/efficientml/atrw"
OUT="/home/deploy/efficientml/tiger_embed"; os.makedirs(OUT, exist_ok=True)
EMB_TENSOR="/Flatten_output_0"; SEED=0

def add_embedding_output():
    m=onnx.load(M2)
    have={o.name for o in m.graph.output}
    if EMB_TENSOR not in have:
        vi=onnx.helper.ValueInfoProto(); vi.name=EMB_TENSOR
        m.graph.output.append(vi)
    onnx.save(m, M2E)
    print("[graph] wrote", M2E, "outputs=", [o.name for o in m.graph.output])

def make_sess(p):
    so=ort.SessionOptions(); so.intra_op_num_threads=4
    return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])

def embed_pixels(sess, pixels, log=""):
    out=[]; t0=time.time(); n=len(pixels)
    for i in range(n):
        x=normalise(pixels[i], CFG)[None].astype(np.float32)
        r=sess.run([EMB_TENSOR], {"input":x})[0]
        out.append(r.reshape(-1))
        if i%1000==0: print(f"  {log} {i}/{n} {time.time()-t0:.0f}s", flush=True)
    e=np.asarray(out, np.float32)
    e/=np.linalg.norm(e,axis=1,keepdims=True)+1e-12
    return e

def embed_files(sess, paths, log=""):
    out=[]; t0=time.time(); n=len(paths)
    for i,p in enumerate(paths):
        x,_=preprocess_file(p, CFG); x=x[None].astype(np.float32)
        r=sess.run([EMB_TENSOR], {"input":x})[0]
        out.append(r.reshape(-1))
        if i%1000==0: print(f"  {log} {i}/{n} {time.time()-t0:.0f}s", flush=True)
    e=np.asarray(out, np.float32)
    e/=np.linalg.norm(e,axis=1,keepdims=True)+1e-12
    return e

def f2(p,r):
    b2=4.0; d=b2*p+r; return 0.0 if d==0 else (1+b2)*p*r/d
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

def main():
    add_embedding_output()
    sess=make_sess(M2E)
    # sanity: logits still present & unchanged shape
    print("[sess] outputs:", [o.name for o in sess.get_outputs()])

    # CCT negatives from cache pixels (same as experiment)
    cct=np.load(f"{ATRW}/emb/cct_cisvalclean.npz", allow_pickle=True)
    neg_lab=cct["labels"].astype(str); neg_ids=cct["ids"].astype(str)
    pix=np.load(f"{REPO}/data/cache/cis_val_clean-256x192/pixels.npy")
    neg=embed_pixels(sess, pix, "cct")
    # ATRW detection positives
    det_paths=np.load(f"{ATRW}/emb/atrw_detection.npz", allow_pickle=True)["paths"].astype(str)
    pos=embed_files(sess, list(det_paths), "atrw-det")
    np.savez(f"{OUT}/m2_emb.npz", neg=neg, neg_lab=neg_lab, pos=pos)

    # same split as experiment
    rng=np.random.default_rng(SEED)
    nval,ntest=halves(neg_lab,SEED); neg_val,neg_test=neg[nval],neg[ntest]
    P=len(pos);perm=rng.permutation(P);npool=int(P*.6);nv=int(P*.2)
    pool=perm[:npool];pval=perm[npool:npool+nv];ptest=perm[npool+nv:]
    pos_val,pos_test=pos[pval],pos[ptest]

    # A2 linear head at K=full pool (deployment: use all available tiger support)
    X=np.concatenate([pos[pool],neg_val]); y=np.concatenate([np.ones(len(pool)),np.zeros(len(neg_val))])
    clf=LogisticRegression(max_iter=5000,class_weight="balanced",C=1.0).fit(X,y)
    sp_v=clf.decision_function(pos_val);sn_v=clf.decision_function(neg_val)
    sp_t=clf.decision_function(pos_test);sn_t=clf.decision_function(neg_test)
    tff=calib_ff(sn_v,.05); tf2=calib_f2(sp_v,sn_v)
    mff=m_at(sp_t,sn_t,tff); mf2=m_at(sp_t,sn_t,tf2)
    yy=np.concatenate([np.ones_like(sp_t),np.zeros_like(sn_t)]);ss=np.concatenate([sp_t,sn_t])
    auc=float(roc_auc_score(yy,ss));ap=float(average_precision_score(yy,ss))
    print(f"[M2-space A2 head] AUC={auc:.4f} AP={ap:.4f} F2opt={mf2['f2']:.3f} "
          f"FF@5%: rec={mff['recall']:.3f} prec={mff['precision']:.3f} fpr={mff['fpr']:.3f}")

    # export deployable head: w[1280], b, threshold (FF@5%)
    w=clf.coef_[0].astype(np.float64); b=float(clf.intercept_[0])
    head=dict(target="tiger", source="ATRW", emb_tensor=EMB_TENSOR, emb_l2norm=True,
              method="A2_linear_probe_M2space", dim=int(len(w)),
              weight=w.tolist(), bias=b, threshold=float(tff),
              calib="largest threshold with val false-fire<=5%",
              metrics_test=dict(auc=auc, ap=ap, f2opt=mf2, ff5=mff),
              n_pos_support=int(npool), n_neg_val=int(len(neg_val)),
              n_pos_test=int(len(pos_test)), n_neg_test=int(len(neg_test)))
    json.dump(head, open(f"{OUT}/tiger_head_M2.json","w"), indent=2)
    print("[export] tiger_head_M2.json  dim", len(w), "threshold", round(tff,4))
    print("T1 DONE")

if __name__=="__main__": main()
