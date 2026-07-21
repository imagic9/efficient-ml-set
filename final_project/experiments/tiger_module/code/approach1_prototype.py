#!/usr/bin/env python3
"""Approach 1 — few-shot feature-space prototype (NO gradient training).

Product idea: "add a new target animal (tiger) with N example images and have the
deployed frozen-backbone system recognise it" — without retraining anything.

Method:
  prototype p = L2-normalise(mean of L2-normed embeddings of K tiger support images)
  score(x)    = cosine(embed(x), p)
  decide      = score > threshold  (threshold calibrated on a validation split)

Backbone: the deployed M0 MobileNetV2, frozen. Embeddings precomputed (emb_build.py).
Positives: ATRW tigers. Negatives: CCT cis_val_clean frames (the camera-trap background
the device normally sees), with per-class labels so we can separate an honest
"appearance" signal (tiger vs other cats) from a cheap "domain" signal (tiger vs empty).
"""
from __future__ import annotations
import os, json, numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

EMB="/home/deploy/efficientml/atrw/emb"
OUT="/home/deploy/efficientml/atrw/results_a1"; os.makedirs(OUT, exist_ok=True)
SEED=0; rng=np.random.default_rng(SEED)

def f2(prec, rec):
    if prec<=0 and rec<=0: return 0.0
    b2=4.0
    d=b2*prec+rec
    return 0.0 if d==0 else (1+b2)*prec*rec/d

def metrics_at_threshold(scores_pos, scores_neg, t):
    tp=(scores_pos> t).sum(); fn=(scores_pos<=t).sum()
    fp=(scores_neg> t).sum(); tn=(scores_neg<=t).sum()
    rec = tp/max(tp+fn,1); prec= tp/max(tp+fp,1); fpr=fp/max(fp+tn,1)
    return dict(precision=float(prec), recall=float(rec), fpr=float(fpr), f2=float(f2(prec,rec)),
                tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn))

def calib_f2opt(pos_val, neg_val):
    cand=np.unique(np.concatenate([pos_val, neg_val]))
    best_t, best=cand[0], -1
    for t in cand:
        m=metrics_at_threshold(pos_val, neg_val, t)
        if m["f2"]>best: best=m["f2"]; best_t=float(t)
    return best_t

def calib_falsefire(neg_val, budget=0.05):
    # largest threshold whose val false-fire <= budget = (1-budget) quantile of neg scores
    return float(np.quantile(neg_val, 1.0-budget))

def stratified_halves(labels, seed):
    r=np.random.default_rng(seed); a=[]; b=[]
    for lab in set(labels):
        idx=np.where(labels==lab)[0]; r.shuffle(idx)
        h=len(idx)//2; a+=idx[:h].tolist(); b+=idx[h:].tolist()
    return np.array(sorted(a)), np.array(sorted(b))

def run(pos_name):
    cct=np.load(f"{EMB}/cct_cisvalclean.npz", allow_pickle=True)
    neg=cct["emb"]; neg_lab=cct["labels"].astype(str)
    pos=np.load(f"{EMB}/{pos_name}.npz", allow_pickle=True)["emb"]

    # --- negatives: stratified 50/50 val/test ---
    nval_idx, ntest_idx = stratified_halves(neg_lab, SEED)
    neg_val, neg_test = neg[nval_idx], neg[ntest_idx]
    neg_test_lab = neg_lab[ntest_idx]

    # --- positives: support pool (60%) / val (20%) / test (20%) ---
    P=len(pos); perm=rng.permutation(P)
    n_pool=int(P*0.6); n_val=int(P*0.2)
    pool_idx=perm[:n_pool]; pval_idx=perm[n_pool:n_pool+n_val]; ptest_idx=perm[n_pool+n_val:]
    pos_val, pos_test = pos[pval_idx], pos[ptest_idx]

    Ks=[1,5,10,20,50]; R=10
    scaling={}
    for K in Ks:
        rows=[]
        for r in range(R):
            rr=np.random.default_rng(1000*r+K)
            sup=pool_idx[rr.choice(len(pool_idx), size=min(K,len(pool_idx)), replace=False)]
            proto=pos[sup].mean(0); proto/=np.linalg.norm(proto)+1e-12
            sp_val=pos_val@proto; sn_val=neg_val@proto
            sp_test=pos_test@proto; sn_test=neg_test@proto
            t_f2=calib_f2opt(sp_val, sn_val)
            t_ff=calib_falsefire(sn_val, 0.05)
            m_f2=metrics_at_threshold(sp_test, sn_test, t_f2)
            m_ff=metrics_at_threshold(sp_test, sn_test, t_ff)
            y=np.concatenate([np.ones_like(sp_test), np.zeros_like(sn_test)])
            s=np.concatenate([sp_test, sn_test])
            auc=roc_auc_score(y,s); ap=average_precision_score(y,s)
            rows.append(dict(auc=auc, ap=ap, f2opt=m_f2, ff=m_ff))
        agg=lambda f:(float(np.mean([f(x) for x in rows])), float(np.std([f(x) for x in rows])))
        scaling[K]=dict(
            n_support=K, reps=R,
            auc=agg(lambda x:x["auc"]), ap=agg(lambda x:x["ap"]),
            f2opt_f2=agg(lambda x:x["f2opt"]["f2"]),
            f2opt_recall=agg(lambda x:x["f2opt"]["recall"]),
            f2opt_precision=agg(lambda x:x["f2opt"]["precision"]),
            ff_recall=agg(lambda x:x["ff"]["recall"]),
            ff_precision=agg(lambda x:x["ff"]["precision"]),
            ff_fpr=agg(lambda x:x["ff"]["fpr"]),
            ff_f2=agg(lambda x:x["ff"]["f2"]),
        )
        print(f"[{pos_name}] K={K:2d}  AUC={scaling[K]['auc'][0]:.4f}  AP={scaling[K]['ap'][0]:.4f}  "
              f"F2opt={scaling[K]['f2opt_f2'][0]:.3f}  FF@5%: rec={scaling[K]['ff_recall'][0]:.3f} "
              f"prec={scaling[K]['ff_precision'][0]:.3f}", flush=True)

    # --- honest domain-vs-appearance probe at K=10, single fixed prototype ---
    Kp=10; rr=np.random.default_rng(42)
    sup=pool_idx[rr.choice(len(pool_idx), size=Kp, replace=False)]
    proto=pos[sup].mean(0); proto/=np.linalg.norm(proto)+1e-12
    sp_test=pos_test@proto; sn_test=neg_test@proto
    groups={"empty":["empty"],
            "other_animals":[l for l in set(neg_test_lab) if l not in ("empty","car","cat","bobcat")],
            "felids(cat+bobcat)":["cat","bobcat"]}
    probe={}
    for g,labs in groups.items():
        mask=np.isin(neg_test_lab, labs)
        sn=sn_test[mask]
        if len(sn)==0: continue
        y=np.concatenate([np.ones_like(sp_test), np.zeros_like(sn)])
        s=np.concatenate([sp_test, sn])
        probe[g]=dict(n_neg=int(mask.sum()), auc=float(roc_auc_score(y,s)),
                      ap=float(average_precision_score(y,s)),
                      neg_mean=float(sn.mean()), neg_p95=float(np.quantile(sn,0.95)))
    probe["tiger_pos"]=dict(n=int(len(sp_test)), mean=float(sp_test.mean()), p05=float(np.quantile(sp_test,0.05)))
    print(f"[{pos_name}] probe:", json.dumps(probe, indent=0))

    result=dict(positives=pos_name, seed=SEED,
                n_pos_total=int(P), n_pos_pool=int(n_pool), n_pos_val=int(len(pos_val)),
                n_pos_test=int(len(pos_test)),
                n_neg_val=int(len(neg_val)), n_neg_test=int(len(neg_test)),
                neg_test_label_dist=dict(Counter(neg_test_lab.tolist())),
                scaling=scaling, domain_probe=probe)
    json.dump(result, open(f"{OUT}/result_{pos_name}.json","w"), indent=2)

    # figures use the fixed K=10 prototype for curves
    y=np.concatenate([np.ones_like(sp_test), np.zeros_like(sn_test)])
    s=np.concatenate([sp_test, sn_test])
    fpr,tpr,_=roc_curve(y,s); prec,rec,_=precision_recall_curve(y,s)
    return result, (Ks, scaling), (sp_test, sn_test, neg_test_lab, fpr, tpr, prec, rec)

def make_figs(det, reid):
    (rdet, (Ks,sdet), curves_det)=det
    (rreid,(_, sreid), _)=reid
    # Fig 1: few-shot scaling (AUC + F2opt vs K), detection vs reid
    fig,ax=plt.subplots(1,2, figsize=(11,4.2))
    for name,sc,c in [("detection (цілі кадри)",sdet,"tab:blue"),("reid (обрізані)",sreid,"tab:orange")]:
        auc=[sc[k]["auc"][0] for k in Ks]; aucs=[sc[k]["auc"][1] for k in Ks]
        f2=[sc[k]["f2opt_f2"][0] for k in Ks]
        ax[0].errorbar(Ks,auc,yerr=aucs,marker="o",label=name,color=c)
        ax[1].plot(Ks,f2,marker="s",label=name,color=c)
    ax[0].set_title("Роздільність (ROC-AUC) від к-сті прикладів K"); ax[0].set_xlabel("K прикладів тигра (support)"); ax[0].set_ylabel("ROC-AUC"); ax[0].set_xscale("log"); ax[0].set_xticks(Ks); ax[0].set_xticklabels(Ks); ax[0].grid(alpha=.3); ax[0].legend(); ax[0].set_ylim(0.5,1.01)
    ax[1].set_title("Якість спрацювання (F2) від K"); ax[1].set_xlabel("K прикладів тигра (support)"); ax[1].set_ylabel("F2 (оптим. поріг)"); ax[1].set_xscale("log"); ax[1].set_xticks(Ks); ax[1].set_xticklabels(Ks); ax[1].grid(alpha=.3); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_fewshot_scaling.png", dpi=130); plt.close(fig)

    # Fig 2: ROC + PR (detection, K=10)
    sp,sn,lab,fpr,tpr,prec,rec=curves_det
    fig,ax=plt.subplots(1,2, figsize=(11,4.2))
    ax[0].plot(fpr,tpr,color="tab:blue"); ax[0].plot([0,1],[0,1],"--",color="gray")
    ax[0].set_title(f"ROC — тигр проти фону CCT (K=10, AUC={rdet['scaling'][10]['auc'][0]:.3f})"); ax[0].set_xlabel("Хибні спрацювання (FPR)"); ax[0].set_ylabel("Повнота (TPR)"); ax[0].grid(alpha=.3)
    ax[1].plot(rec,prec,color="tab:green")
    ax[1].set_title(f"Precision–Recall (AP={rdet['scaling'][10]['ap'][0]:.3f})"); ax[1].set_xlabel("Повнота (recall)"); ax[1].set_ylabel("Точність (precision)"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_roc_pr.png", dpi=130); plt.close(fig)

    # Fig 3: score histograms tiger vs negatives (by group)
    fig,ax=plt.subplots(figsize=(8,4.4))
    ax.hist(sp, bins=40, alpha=.6, density=True, label="тигр (ATRW)", color="tab:red")
    for labs,c,nm in [(["empty"],"tab:gray","порожній кадр"),
                      (["cat","bobcat"],"tab:purple","коти CCT (cat/bobcat)")]:
        m=np.isin(lab,labs); ax.hist(sn[m], bins=40, alpha=.5, density=True, label=nm, color=c)
    ax.set_title("Розподіл косинусної близькості до прототипа тигра"); ax.set_xlabel("cosine(embed, прототип тигра)"); ax.set_ylabel("щільність"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_score_hist.png", dpi=130); plt.close(fig)

    # Fig 4: domain-vs-appearance probe (AUC per negative group)
    pr=rdet["domain_probe"]; groups=[g for g in pr if g!="tiger_pos"]
    aucs=[pr[g]["auc"] for g in groups]
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar(range(len(groups)), aucs, color=["tab:gray","tab:green","tab:purple"][:len(groups)])
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, rotation=15); ax.set_ylim(0.5,1.01)
    ax.axhline(1.0, ls="--", color="k", alpha=.3)
    for i,a in enumerate(aucs): ax.text(i,a+0.005,f"{a:.3f}",ha="center")
    ax.set_title("Чесна перевірка: тигр проти РІЗНИХ негативів (K=10)"); ax.set_ylabel("ROC-AUC")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_domain_probe.png", dpi=130); plt.close(fig)
    print("figures written ->", OUT)

if __name__=="__main__":
    det=run("atrw_detection")
    reid=run("atrw_reid")
    make_figs(det, reid)
    print("APPROACH 1 DONE")
