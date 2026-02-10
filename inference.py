import torch
import cv2
from models.fusion_net import RGBIRFusionNet

device = "cuda" if torch.cuda.is_available() else "cpu"
model = RGBIRFusionNet().to(device)
model.load_state_dict(torch.load("fusion_model.pth"))
model.eval()

rgb = cv2.imread("sample_rgb.png")
ir = cv2.imread("sample_ir.png", cv2.IMREAD_GRAYSCALE)

rgb = torch.from_numpy(rgb).permute(2,0,1).float().unsqueeze(0) / 255.0
ir = cv2.cvtColor(ir, cv2.COLOR_GRAY2RGB)
ir = torch.from_numpy(ir).permute(2,0,1).float().unsqueeze(0) / 255.0

with torch.no_grad():
    output = model(rgb.to(device), ir.to(device))

print("Inference done")
