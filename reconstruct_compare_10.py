import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from dataset import FastMRISliceDataset
from model import ResidualAttentionUNet
from utils import compute_psnr

# ---------------- PATHS ----------------
ROOT = os.path.expanduser("~/Anandhakrishnan P V")
DATA = os.path.join(ROOT, "data", "val")
OUT  = os.path.join(ROOT, "final_result", "img_final")
CKPT = os.path.join(ROOT, "final_result", "checkpoints", "best_checkpoint.pth")

os.makedirs(OUT, exist_ok=True)

# ---------------- LOAD DATA ----------------
val_files = [
    os.path.join(DATA, f)
    for f in os.listdir(DATA)
    if f.endswith(".h5")
]

dataset = FastMRISliceDataset(val_files)

# ---------------- LOAD MODEL ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResidualAttentionUNet().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# ---------------- SELECT 10 SAMPLES ----------------
indices = random.sample(range(len(dataset)), 10)

for idx_i, idx in enumerate(indices, 1):
    x, y = dataset[idx]

    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device))[0, 0].cpu()

    # -------- PSNR COMPUTATION --------
    psnr_input = compute_psnr(x[0], y[0]).item()
    psnr_pred  = compute_psnr(pred, y[0]).item()

    # -------- PLOTTING --------
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(x[0], cmap="gray")
    axs[0].set_title(f"Input (Zero-filled)\nPSNR: {psnr_input:.2f} dB")

    axs[1].imshow(pred, cmap="gray")
    axs[1].set_title(f"Reconstruction\nPSNR: {psnr_pred:.2f} dB")

    axs[2].imshow(y[0], cmap="gray")
    axs[2].set_title("Ground Truth")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUT, f"compare_{idx_i:02d}.jpg"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

print("✅ 10 comparison images saved successfully")
