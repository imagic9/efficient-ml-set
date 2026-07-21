#!/usr/bin/env python3
"""Build frozen-backbone embeddings for CCT negatives and ATRW tigers.

Reuses the EXACT project preprocessing (same letterbox 256x192, same ImageNet
normalisation) and the deployed M0 MobileNetV2 backbone (features -> global avg
pool -> 1280-d, L2-normalised). No gradients, no training.
"""
from __future__ import annotations
import os, sys, json, hashlib, time
import numpy as np, torch

ROOT = "/home/deploy/efficientml/efficient-ml-set/final_project"
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from wildlife_trigger.data.preprocess import PreprocessConfig, normalise, preprocess_file
from wildlife_trigger.models.mobilenet import build_mobilenet_v2

CKPT = "results/training/c2/c2_m0_fp32_seed42_20260716T061203Z/best.pt"
OUT  = "/home/deploy/efficientml/atrw/emb"
os.makedirs(OUT, exist_ok=True)
DEV  = "cuda" if torch.cuda.is_available() else "cpu"
CFG  = PreprocessConfig()  # default 256x192

def load_backbone():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model"]; names = ck.get("class_names")
    model = build_mobilenet_v2(num_classes=len(names) if names else 16, pretrained=False)
    model.load_state_dict(sd)
    model.eval().to(DEV)
    print(f"[backbone] loaded {CKPT}  classes={names}")
    return model, names

@torch.inference_mode()
def embed_batch(model, nchw_np):
    x = torch.from_numpy(nchw_np).to(DEV)
    feat = model.features(x)                       # (B,1280,6,8)
    v = torch.nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B,1280)
    v = torch.nn.functional.normalize(v, dim=1)    # L2
    return v.float().cpu().numpy()

def embed_from_pixels(model, pixels, bs=256, log=""):
    out=[]; n=len(pixels); t0=time.time()
    for i in range(0, n, bs):
        chunk = pixels[i:i+bs]
        nchw = np.stack([normalise(chunk[j], CFG) for j in range(len(chunk))])
        out.append(embed_batch(model, nchw))
        if i % (bs*8)==0: print(f"  {log} {i}/{n} {time.time()-t0:.0f}s", flush=True)
    return np.concatenate(out)

def embed_from_files(model, paths, bs=128, log=""):
    out=[]; n=len(paths); t0=time.time()
    for i in range(0, n, bs):
        batch=[]
        for p in paths[i:i+bs]:
            nchw,_ = preprocess_file(p, CFG)
            batch.append(nchw)
        out.append(embed_batch(model, np.stack(batch)))
        if i % (bs*4)==0: print(f"  {log} {i}/{n} {time.time()-t0:.0f}s", flush=True)
    return np.concatenate(out)

def main():
    model, names = load_backbone()

    # --- CCT negatives: cis_val_clean cache (uint8 letterbox) + manifest labels ---
    pix = np.load("data/cache/cis_val_clean-256x192/pixels.npy")  # (3214,192,256,3)
    recs=[json.loads(l) for l in open("data/manifests/cis_val_clean.jsonl")]
    assert len(recs)==len(pix), (len(recs), len(pix))
    ids=[r["image_id"] for r in recs]
    labels=[r.get("primary_label") or (r.get("labels") or ["?"])[0] for r in recs]
    # alignment sanity: sha256 of ordered image_ids vs meta
    meta=json.load(open("data/cache/cis_val_clean-256x192/meta.json"))
    for sep in ("\n",",",""):
        h=hashlib.sha256(sep.join(ids).encode()).hexdigest()
        if h==meta.get("image_id_order_sha256"):
            print(f"[cct] alignment CONFIRMED via sep={sep!r}"); break
    else:
        print("[cct] alignment sha not matched by simple joins; trusting manifest order")
    from collections import Counter
    print("[cct] label dist:", Counter(labels).most_common())
    cct_emb = embed_from_pixels(model, pix, log="cct")
    np.savez(os.path.join(OUT,"cct_cisvalclean.npz"),
             emb=cct_emb, labels=np.array(labels), ids=np.array(ids))
    print("[cct] saved", cct_emb.shape)

    # --- ATRW tiger positives: detection whole frames + reid crops ---
    det_dir="/home/deploy/efficientml/atrw/trainval"
    det=sorted(os.path.join(det_dir,f) for f in os.listdir(det_dir) if f.endswith(".jpg"))
    det_emb=embed_from_files(model, det, log="atrw-det")
    np.savez(os.path.join(OUT,"atrw_detection.npz"),
             emb=det_emb, paths=np.array(det))
    print("[atrw-det] saved", det_emb.shape)

    reid_dir="/home/deploy/efficientml/atrw/train"
    reid=sorted(os.path.join(reid_dir,f) for f in os.listdir(reid_dir) if f.endswith(".jpg"))
    reid_emb=embed_from_files(model, reid, log="atrw-reid")
    np.savez(os.path.join(OUT,"atrw_reid.npz"),
             emb=reid_emb, paths=np.array(reid))
    print("[atrw-reid] saved", reid_emb.shape)
    print("ALL DONE ->", OUT)

if __name__=="__main__":
    main()
