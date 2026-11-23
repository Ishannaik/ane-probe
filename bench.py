"""SigLIP2 vision ONNX: CPU vs CoreML EP (CPUAndNeuralEngine) on this runner."""
import time
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

path = hf_hub_download("Ishannaik/siglip2-base-onnx", "vision_encoder_dynamic.onnx")
x = np.random.default_rng(0).standard_normal((32, 3, 224, 224)).astype(np.float32)
for name, provs in [
    ("cpu", ["CPUExecutionProvider"]),
    ("coreml_ane", [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndNeuralEngine"}), "CPUExecutionProvider"]),
    ("coreml_all", [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}), "CPUExecutionProvider"]),
]:
    t0 = time.time()
    s = ort.InferenceSession(path, providers=provs)
    n = s.get_inputs()[0].name
    s.run(None, {n: x})
    first = time.time() - t0
    t1 = time.time()
    for _ in range(5):
        s.run(None, {n: x})
    print(f"{name}: load+first {first:.1f}s, batch32 {(time.time() - t1) / 5:.3f}s")
