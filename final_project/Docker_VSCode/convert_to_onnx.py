#!/usr/bin/env python3
import torch
import torchvision.models as models

def convert_mobilenet_to_onnx():
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model,
        dummy_input,
        "mobilenet_v2.onnx",
        export_params=True,
        opset_version=9,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )
    
    print("MobileNet V2 model successfully converted to ONNX format: mobilenet_v2.onnx")

if __name__ == "__main__":
    convert_mobilenet_to_onnx()
