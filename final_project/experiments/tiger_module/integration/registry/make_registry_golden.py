import os, sys, json, numpy as np, onnxruntime as ort
REPO="/home/deploy/efficientml/efficient-ml-set/final_project"
sys.path.insert(0, os.path.join(REPO,"src")); os.chdir(REPO)
from wildlife_trigger.data.preprocess import PreprocessConfig, normalise, preprocess_file
CFG=PreprocessConfig(); TE="/home/deploy/efficientml/tiger_embed"
reg=json.load(open(f"{TE}/target_registry.json"))
det=np.load(f"{TE}/../atrw/emb/atrw_detection.npz", allow_pickle=True)["paths"].astype(str)
cct=np.load(f"{TE}/../atrw/emb/cct_cisvalclean.npz", allow_pickle=True)
labs=cct["labels"].astype(str); pix=np.load(f"{REPO}/data/cache/cis_val_clean-256x192/pixels.npy")

# one demo image per category (use a late index to avoid fit overlap where possible)
def cct_img(lbl): 
    idx=np.where(labs==lbl)[0]; return normalise(pix[idx[-1]], CFG).astype(np.float32)
samples={"tiger":preprocess_file(str(det[100]),CFG)[0].astype(np.float32),
         "bobcat":cct_img("bobcat"),"coyote":cct_img("coyote"),
         "raccoon":cct_img("raccoon"),"empty":cct_img("empty")}

so=ort.SessionOptions(); so.intra_op_num_threads=1
sess=ort.InferenceSession(f"{TE}/models/M2_plus.onnx",so,providers=["CPUExecutionProvider"])
W=[(t["name"],np.array(t["weight"]),t["bias"],t["threshold"]) for t in reg["targets"]]
ref={}
print(f"{'image':9s} | " + " ".join(f"{n:>8s}" for n,_,_,_ in W))
for name,x in samples.items():
    x.tofile(f"{TE}/reg_{name}.bin")
    emb=sess.run(["/Flatten_output_0"],{"input":x[None]})[0].reshape(-1)
    en=emb/(np.linalg.norm(emb)+1e-12)
    row={}; cells=[]
    for tn,w,b,thr in W:
        s=float(w@en+b); fired=bool(s>thr); row[tn]=dict(score=s,fired=fired)
        cells.append(f"{'FIRE' if fired else '  .':>8s}")
    ref[name]=row
    print(f"{name:9s} | " + " ".join(cells))
json.dump(ref, open(f"{TE}/reg_ref.json","w"), indent=2)
print("golden + reference written")
