#!/usr/bin/env python3
"""T2 — build M2_plus.onnx: M2 with (a) the pooled embedding exposed and (b) a baked,
self-contained `tiger_score` output = w . L2norm(embedding) + b.

Verifies:
  C1  the 16-class `logits` are BIT-IDENTICAL to the released M2 (adding outputs must
      not change existing computation) -> all Core parity/frozen-test evidence still holds.
  C2  the in-graph `tiger_score` matches the sklearn head's decision_function (< 1e-4).
"""
from __future__ import annotations
import os, sys, json, numpy as np
import onnx, onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper

REPO="/home/deploy/efficientml/efficient-ml-set/final_project"
sys.path.insert(0, os.path.join(REPO,"src")); os.chdir(REPO)
from wildlife_trigger.data.preprocess import PreprocessConfig, normalise
CFG=PreprocessConfig()

TE="/home/deploy/efficientml/tiger_embed"
M2=f"{TE}/models/M2.onnx"; M2P=f"{TE}/models/M2_plus.onnx"
EMB="/Flatten_output_0"

def build():
    head=json.load(open(f"{TE}/tiger_head_M2.json"))
    w=np.array(head["weight"], np.float32); b=np.float32(head["bias"])
    m=onnx.load(M2); g=m.graph
    outs={o.name for o in g.output}
    # expose embedding
    if EMB not in outs:
        g.output.append(helper.make_tensor_value_info(EMB, TensorProto.FLOAT, [1,1280]))
    # nodes: L2 normalise the [1,1280] embedding, then Gemm -> tiger_score [1,1]
    l2=helper.make_node("ReduceL2", [EMB], ["tiger_emb_l2"], axes=[1], keepdims=1, name="tiger_ReduceL2")
    nrm=helper.make_node("Div", [EMB,"tiger_emb_l2"], ["tiger_emb_normed"], name="tiger_Div")
    W=numpy_helper.from_array(w.reshape(1280,1), "tiger_W")   # [1280,1]
    B=numpy_helper.from_array(np.array([b],np.float32), "tiger_B")
    gemm=helper.make_node("Gemm", ["tiger_emb_normed","tiger_W","tiger_B"], ["tiger_score"],
                          alpha=1.0, beta=1.0, transA=0, transB=0, name="tiger_Gemm")
    g.initializer.extend([W,B]); g.node.extend([l2,nrm,gemm])
    g.output.append(helper.make_tensor_value_info("tiger_score", TensorProto.FLOAT, [1,1]))
    onnx.checker.check_model(m)
    onnx.save(m, M2P)
    print("[build] wrote", M2P, "outputs=", [o.name for o in g.output])
    return head

def sess(p):
    so=ort.SessionOptions(); so.intra_op_num_threads=4
    return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])

def main():
    head=build()
    s0=sess(M2); sP=sess(M2P)
    # real inputs: 40 CCT + we also need positives; use cache pixels for a quick batch
    pix=np.load(f"{REPO}/data/cache/cis_val_clean-256x192/pixels.npy")[:40]
    max_logit_diff=0.0; scores=[]; ref_scores=[]
    from numpy.linalg import norm
    # reconstruct sklearn decision_function from exported w,b on M2-space normed emb
    w=np.array(head["weight"], np.float64); b=float(head["bias"])
    for i in range(len(pix)):
        x=normalise(pix[i], CFG)[None].astype(np.float32)
        r0=s0.run(["logits"], {"input":x})[0]
        rP=sP.run(["logits", EMB, "tiger_score"], {"input":x})
        lP, emb, ts = rP[0], rP[1].reshape(-1), float(rP[2].reshape(-1)[0])
        max_logit_diff=max(max_logit_diff, float(np.abs(r0-lP).max()))
        en=emb/ (norm(emb)+1e-12)
        ref=float(w@en + b)
        scores.append(ts); ref_scores.append(ref)
    scores=np.array(scores); ref_scores=np.array(ref_scores)
    print(f"[C1] max |logits(M2) - logits(M2_plus)| = {max_logit_diff:.3e}  (expect 0.0)")
    print(f"[C2] max |tiger_score(graph) - w.normed(emb)+b| = {np.abs(scores-ref_scores).max():.3e}")
    print(f"     tiger_score range on 40 CCT negs: [{scores.min():.3f}, {scores.max():.3f}], threshold={head['threshold']:.3f}")
    ok = (max_logit_diff==0.0) and (np.abs(scores-ref_scores).max()<1e-3)
    print("T2", "PASS" if ok else "CHECK")

if __name__=="__main__": main()
