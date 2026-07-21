# Build M2_plus_lean.onnx: tiger_score baked, but the embedding is INTERNAL (not a
# graph output) so ORT can still fuse the classifier tail.
import onnx, numpy as np, json
from onnx import helper, TensorProto, numpy_helper
TE="/home/deploy/efficientml/tiger_embed"
head=json.load(open(f"{TE}/tiger_head_M2.json"))
w=np.array(head["weight"],np.float32); b=np.float32(head["bias"]); EMB="/Flatten_output_0"
m=onnx.load(f"{TE}/models/M2.onnx"); g=m.graph
l2=helper.make_node("ReduceL2",[EMB],["tiger_emb_l2"],axes=[1],keepdims=1,name="tiger_ReduceL2")
nrm=helper.make_node("Div",[EMB,"tiger_emb_l2"],["tiger_emb_normed"],name="tiger_Div")
W=numpy_helper.from_array(w.reshape(1280,1),"tiger_W"); B=numpy_helper.from_array(np.array([b],np.float32),"tiger_B")
gemm=helper.make_node("Gemm",["tiger_emb_normed","tiger_W","tiger_B"],["tiger_score"],alpha=1.0,beta=1.0,name="tiger_Gemm")
g.initializer.extend([W,B]); g.node.extend([l2,nrm,gemm])
g.output.append(helper.make_tensor_value_info("tiger_score",TensorProto.FLOAT,[1,1]))
onnx.checker.check_model(m); onnx.save(m,f"{TE}/models/M2_plus_lean.onnx")
print("wrote M2_plus_lean.onnx outputs=",[o.name for o in g.output])
