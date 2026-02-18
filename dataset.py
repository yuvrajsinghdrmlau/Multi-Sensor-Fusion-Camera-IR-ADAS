import os
import cv2
import torch
from torch.utils.data import Dataset

class RGBIRDataset(Dataset):
    def __init__(self, rgb_dir, ir_dir, transform=None):
        self.rgb_files = sorted(os.listdir(rgb_dir))
        self.ir_files = sorted(os.listdir(ir_dir))
        self.rgb_dir = rgb_dir
        self.ir_dir = ir_dir
        self.transform = transform
#de
    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb = cv2.imread(os.path.join(self.rgb_dir, self.rgb_files[idx]))
        ir = cv2.imread(os.path.join(self.ir_dir, self.ir_files[idx]), cv2.IMREAD_GRAYSCALE)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        ir = cv2.cvtColor(ir, cv2.COLOR_GRAY2RGB)

        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        ir = torch.from_numpy(ir).permute(2, 0, 1).float() / 255.0

        return rgb, ir
