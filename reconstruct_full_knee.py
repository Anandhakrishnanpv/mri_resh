import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import FastMRISliceDataset
from model import ResidualAttentionUNet

# ================= PATHS =================
ROOT = os.path.expanduser("~/Anandhakrishnan P V")
DATA = os.path.join(ROOT, "data", "val")
OUT  = os.path.join(ROOT, "final_research_values", "full_knee")

CKPT = os.path.join(
    ROOT,
    "final_result",
    "checkpoints",
    "best_checkpoint.pth"
)

os.makedirs(OUT, exist_ok=True)

# ================= LOAD DATA =================
val_files = sorted([
    os.path.join(DATA, f)
    for f in os.listdir(DATA)
    if f.endswith(".h5")
])

dataset = FastMRISliceDataset(val_files)

# ================= LOAD MODEL =================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResidualAttentionUNet().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# ================= COLLECT SLICES =================
recon_slices = []
gt_slices = []

with torch.no_grad():
    for i in range(len(dataset)):
        x, y = dataset[i]
        pred = model(x.unsqueeze(0).to(device))[0, 0].cpu()

        recon_slices.append(pred.numpy())
        gt_slices.append(y[0].numpy())

# ================= STACK FULL KNEE =================
recon_stack = np.stack(recon_slices, axis=0)
gt_stack    = np.stack(gt_slices, axis=0)

# Normalize globally for clean visualization
def normalize(vol):
    return (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

recon_stack = normalize(recon_stack)
gt_stack    = normalize(gt_stack)

# ================= CREATE MOSAIC =================
def make_mosaic(volume, cols=10):
    slices, H, W = volume.shape
    rows = int(np.ceil(slices / cols))

    canvas = np.zeros((rows * H, cols * W))
    for i in range(slices):
        r = i // cols
        c = i % cols
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = volume[i]

    return canvas

recon_mosaic = make_mosaic(recon_stack, cols=10)
gt_mosaic    = make_mosaic(gt_stack, cols=10)

# ================= SAVE IMAGES =================
plt.figure(figsize=(12, 12))
plt.imshow(recon_mosaic, cmap="gray")
plt.title("Full Knee Reconstruction (Model Output)")
plt.axis("off")
plt.savefig(os.path.join(OUT, "full_knee_reconstruction.jpg"), dpi=300)
plt.close()

plt.figure(figsize=(12, 12))
plt.imshow(gt_mosaic, cmap="gray")
plt.title("Full Knee Ground Truth")
plt.axis("off")
plt.savefig(os.path.join(OUT, "full_knee_ground_truth.jpg"), dpi=300)
plt.close()

print("✅ Full knee reconstructed images saved")
print(f"📁 Location: {OUT}")
