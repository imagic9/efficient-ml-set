#!/usr/bin/env python3
"""Approach 2 — head-only training (linear probe on the FROZEN backbone).

Same frozen MobileNetV2 embeddings as Approach 1. Instead of a prototype, we train a
small linear classifier (logistic regression = one trainable Linear head) on:
  K tiger positives (the new animal) + the available CCT background negatives.
This is exactly "freeze the backbone, train only the head" reduced to its essence,
and it is the realistic product setup: the device already has lots of background frames,
you supply a few examples of the new animal.
"""
from __future__ import annotations
import os, json, numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

EMB="/home/deploy/efficientml/atrw/emb"
OUT="/home/deploy/efficientml/atrw/results_a2"; os.makedirs(OUT, exist_ok=True)
SEED=0; rng=np.random.default_rng(SEED)

def f2(p,r):
    b2=4.0; d=b2*p+r
    return 0.0 if d==0 else (1+b2)*p*r/d
def m_at(sp,sn,t):
    tp=(sp>t).sum();fn=(sp<=t).sum();fp=(sn>t).sum();tn=(sn<=t).sum()
    rec=tp/max(tp+fn,1);prec=tp/max(tp+fp,1);fpr=fp/max(fp+tn,1)
    return dict(precision=float(prec),recall=float(rec),fpr=float(fpr),f2=float(f2(prec,rec)))
def calib_f2(sp,sn):
    c=np.unique(np.concatenate([sp,sn]));bt,bb=c[0],-1
    for t in c:
        m=m_at(sp,sn,t)
        if m["f2"]>bb: bb=m["f2"];bt=float(t)
    return bt
def calib_ff(sn,b=0.05): return float(np.quantile(sn,1-b))
def halves(lab,seed):
    r=np.random.default_rng(seed);a=[];b=[]
    for L in set(lab):
        idx=np.where(lab==L)[0];r.shuffle(idx);h=len(idx)//2;a+=idx[:h].tolist();b+=idx[h:].tolist()
    return np.array(sorted(a)),np.array(sorted(b))

def run(pos_name):
    cct=np.load(f"{EMB}/cct_cisvalclean.npz",allow_pickle=True)
    neg=cct["emb"];neg_lab=cct["labels"].astype(str)
    pos=np.load(f"{EMB}/{pos_name}.npz",allow_pickle=True)["emb"]
    nval,ntest=halves(neg_lab,SEED)
    neg_val,neg_test=neg[nval],neg[ntest]
    P=len(pos);perm=rng.permutation(P);npool=int(P*.6);nv=int(P*.2)
    pool=perm[:npool];pval=perm[npool:npool+nv];ptest=perm[npool+nv:]
    pos_val,pos_test=pos[pval],pos[ptest]
    Ks=[1,5,10,20,50];R=10;scaling={}
    for K in Ks:
        rows=[]
        for r in range(R):
            rr=np.random.default_rng(1000*r+K)
            sup=pool[rr.choice(len(pool),size=min(K,len(pool)),replace=False)]
            X=np.concatenate([pos[sup],neg_val]); y=np.concatenate([np.ones(len(sup)),np.zeros(len(neg_val))])
            clf=LogisticRegression(max_iter=2000,class_weight="balanced",C=1.0).fit(X,y)
            sp_v=clf.decision_function(pos_val);sn_v=clf.decision_function(neg_val)
            sp_t=clf.decision_function(pos_test);sn_t=clf.decision_function(neg_test)
            tf2=calib_f2(sp_v,sn_v);tff=calib_ff(sn_v,.05)
            mf2=m_at(sp_t,sn_t,tf2);mff=m_at(sp_t,sn_t,tff)
            yy=np.concatenate([np.ones_like(sp_t),np.zeros_like(sn_t)]);ss=np.concatenate([sp_t,sn_t])
            rows.append(dict(auc=roc_auc_score(yy,ss),ap=average_precision_score(yy,ss),f2=mf2,ff=mff))
        agg=lambda f:(float(np.mean([f(x) for x in rows])),float(np.std([f(x) for x in rows])))
        scaling[K]=dict(auc=agg(lambda x:x["auc"]),ap=agg(lambda x:x["ap"]),
            f2opt_f2=agg(lambda x:x["f2"]["f2"]),ff_recall=agg(lambda x:x["ff"]["recall"]),
            ff_precision=agg(lambda x:x["ff"]["precision"]),ff_fpr=agg(lambda x:x["ff"]["fpr"]),
            ff_f2=agg(lambda x:x["ff"]["f2"]))
        print(f"[A2 {pos_name}] K={K:2d} AUC={scaling[K]['auc'][0]:.4f} AP={scaling[K]['ap'][0]:.4f} "
              f"F2opt={scaling[K]['f2opt_f2'][0]:.3f} FF@5%: rec={scaling[K]['ff_recall'][0]:.3f} prec={scaling[K]['ff_precision'][0]:.3f}",flush=True)
    json.dump(dict(positives=pos_name,method="linear_probe_head",scaling=scaling),
              open(f"{OUT}/result_{pos_name}.json","w"),indent=2)

if __name__=="__main__":
    run("atrw_detection"); run("atrw_reid"); print("APPROACH 2 DONE")
