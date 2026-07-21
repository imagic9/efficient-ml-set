import os, sys, json, numpy as np, onnxruntime as ort
REPO="/home/deploy/efficientml/efficient-ml-set/final_project"
sys.path.insert(0, os.path.join(REPO,"src")); os.chdir(REPO)
from wildlife_trigger.data.preprocess import PreprocessConfig, normalise, preprocess_file
CFG=PreprocessConfig()
TE="/home/deploy/efficientml/tiger_embed"; M2P=f"{TE}/models/M2_plus.onnx"
head=json.load(open(f"{TE}/tiger_head_M2.json")); THR=head["threshold"]

# pick samples: 1 tiger, 1 CCT empty, 1 CCT bobcat
det=np.load(f"{TE}/../atrw/emb/atrw_detection.npz", allow_pickle=True)["paths"].astype(str)
tiger_path=str(det[0])
cct=np.load(f"{TE}/../atrw/emb/cct_cisvalclean.npz", allow_pickle=True)
labs=cct["labels"].astype(str); ids=cct["ids"].astype(str)
pix=np.load(f"{REPO}/data/cache/cis_val_clean-256x192/pixels.npy")
i_empty=int(np.where(labs=="empty")[0][0]); i_bob=int(np.where(labs=="bobcat")[0][0])

samples={}
# tiger via file
xt,_=preprocess_file(tiger_path, CFG); samples["tiger"]=xt.astype(np.float32)
samples["cct_empty"]=normalise(pix[i_empty], CFG).astype(np.float32)
samples["cct_bobcat"]=normalise(pix[i_bob], CFG).astype(np.float32)

so=ort.SessionOptions(); so.intra_op_num_threads=1
sess=ort.InferenceSession(M2P, so, providers=["CPUExecutionProvider"])
ref={}
for name,x in samples.items():
    xb=x[None]
    x.tofile(f"{TE}/golden_{name}.bin")  # raw float32 NCHW
    o=sess.run(["logits","tiger_score"], {"input":xb})
    logits=o[0].reshape(-1); ts=float(o[1].reshape(-1)[0])
    ref[name]=dict(argmax=int(logits.argmax()), tiger_score=ts,
                   trigger=bool(ts>THR), logits=[float(v) for v in logits])
    print(f"{name:12s} argmax={logits.argmax():2d} tiger_score={ts:+.5f} trigger={ts>THR}")
ref["_threshold"]=THR
json.dump(ref, open(f"{TE}/golden_ref.json","w"), indent=2)
print("threshold", round(THR,4), "-> bin+ref written")
