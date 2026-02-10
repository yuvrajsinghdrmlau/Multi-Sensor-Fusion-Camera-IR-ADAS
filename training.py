import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import RGBIRDataset
from models.fusion_net import RGBIRFusionNet

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = RGBIRDataset(
    rgb_dir="data/rgb",
    ir_dir="data/ir"
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = RGBIRFusionNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(10):
    model.train()
    for rgb, ir in loader:
        rgb, ir = rgb.to(device), ir.to(device)
        target = torch.zeros((rgb.size(0), 1, 7, 7)).to(device)

        pred = model(rgb, ir)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
