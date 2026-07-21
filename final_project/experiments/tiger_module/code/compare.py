#!/usr/bin/env python3
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A="/home/deploy/efficientml/atrw"
a1=json.load(open(f"{A}/results_a1/result_atrw_detection.json"))
a2=json.load(open(f"{A}/results_a2/result_atrw_detection.json"))
a3=json.load(open(f"{A}/results_a3/result_distill.json"))
Ks=[1,5,10,20,50]
def curve(r,key): return [r["scaling"][str(k)][key][0] for k in Ks]

fig,ax=plt.subplots(1,2,figsize=(12,4.6))
ax[0].errorbar(Ks,curve(a1,"auc"),marker="o",label="A1 прототип (0 навчання)",color="tab:blue")
ax[0].errorbar(Ks,curve(a2,"auc"),marker="s",label="A2 лінійна голова",color="tab:green")
ax[0].axhline(a3["student_distilled"]["auc"],ls="--",color="tab:red",
    label=f"A3 дистиляція (0 міток) = {a3['student_distilled']['auc']:.3f}")
ax[0].axhline(a3["teacher_zeroshot"]["auc"],ls=":",color="gray",
    label=f"вчитель zero-shot = {a3['teacher_zeroshot']['auc']:.3f}")
ax[0].set_xscale("log");ax[0].set_xticks(Ks);ax[0].set_xticklabels(Ks)
ax[0].set_title("Роздільність (ROC-AUC) — цілі кадри");ax[0].set_xlabel("K прикладів тигра");ax[0].set_ylabel("ROC-AUC")
ax[0].set_ylim(0.9,1.005);ax[0].grid(alpha=.3);ax[0].legend(fontsize=8)

ax[1].plot(Ks,curve(a1,"ff_recall"),marker="o",label="A1 прототип",color="tab:blue")
ax[1].plot(Ks,curve(a2,"ff_recall"),marker="s",label="A2 лінійна голова",color="tab:green")
ax[1].axhline(a3["student_distilled"]["ff5"]["recall"],ls="--",color="tab:red",label="A3 дистиляція (0 міток)")
ax[1].set_xscale("log");ax[1].set_xticks(Ks);ax[1].set_xticklabels(Ks)
ax[1].set_title("Повнота за бюджету 5% хибних спрацювань");ax[1].set_xlabel("K прикладів тигра");ax[1].set_ylabel("Повнота (recall)")
ax[1].set_ylim(0.6,1.01);ax[1].grid(alpha=.3);ax[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(f"{A}/results_a1/fig5_comparison.png",dpi=130);plt.close(fig)

# compact comparison table json
tab=dict(
 setting="ATRW tiger (detection whole-frames) vs CCT cis_val_clean background; test held out once",
 A1_prototype={f"K{k}":dict(auc=a1["scaling"][str(k)]["auc"][0],f2opt=a1["scaling"][str(k)]["f2opt_f2"][0],
        ff5_recall=a1["scaling"][str(k)]["ff_recall"][0]) for k in Ks},
 A2_head={f"K{k}":dict(auc=a2["scaling"][str(k)]["auc"][0],f2opt=a2["scaling"][str(k)]["f2opt_f2"][0],
        ff5_recall=a2["scaling"][str(k)]["ff_recall"][0]) for k in Ks},
 A3_distill=dict(tiger_labels_used=0,teacher_auc=a3["teacher_zeroshot"]["auc"],
        student_auc=a3["student_distilled"]["auc"],student_f2opt=a3["student_distilled"]["f2opt"]["f2"],
        student_ff5_recall=a3["student_distilled"]["ff5"]["recall"]),
 felid_probe_A1_K10=a1["domain_probe"]["felids(cat+bobcat)"]["auc"],
)
json.dump(tab,open(f"{A}/results_a1/comparison_summary.json","w"),indent=2)
print(json.dumps(tab,indent=2))
