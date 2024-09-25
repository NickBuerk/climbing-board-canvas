import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch.onnx
import torch.nn as nn
from PIL import Image
import numpy as np
import sys


class SamMaskWrapper(nn.Module):
    def __init__(self, mask_generator):
        super(SamMaskWrapper, self).__init__()
        self.mask_generator = mask_generator

    def forward(self, x):
        # Forward pass through the mask generator
        return self.mask_generator.generate(x)  # Adjust based on actual method

# Load the SAM model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = sam_model_registry["vit_b"](checkpoint="/tmp/sam_vit_b_01ec64.pth")
model.to(device=device)
model.eval()  # Set the model to evaluation mode

# Initialize the SamAutomaticMaskGenerator
mask_generator = SamAutomaticMaskGenerator(model)

wrapped_model = SamMaskWrapper(mask_generator)
wrapped_model.to(device)

#dummy_input = torch.randn(4, 1024, 1024).to(device)  # Adjust dimensions as needed
dummy_input = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)

out = wrapped_model.mask_generator.generate(dummy_input)

onnx_model_path = './models/SAM_auto.onnx'

# Export the model
torch.onnx.export(
    wrapped_model,            # Model to export
    dummy_input,              # Dummy input
    onnx_model_path,          # Output file path
    export_params=True,       # Store the trained parameter weights inside the model file
    verbose=True,
    opset_version=11,         # ONNX version to export to
    do_constant_folding=True, # Optimization to fold constants
    input_names=['input'],    # Input name for the ONNX model
    output_names=['output'],  # Output name for the ONNX model
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dynamic axes for batch size
)