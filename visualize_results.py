import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import FastMRISliceDataset
from model import ResidualAttentionUNet
from utils import compute_psnr

# ---------------- PATHS ----------------
ROOT = os.path.expanduser("~/Anandhakrishnan P V")
FINAL = os.path.join(ROOT, "final_result")
LOGS  = os.path.join(FINAL, "logs")
VIS   = os.path.join(FINAL, "visualize")
DATA  = os.path.join(ROOT, "data", "val")
CKPT  = os.path.join(FINAL, "checkpoints", "best_checkpoint.pth")

os.makedirs(VIS, exist_ok=True)

# ======================================================
# 1️⃣ TRAINING CURVES (LOSS / PSNR / SSIM)
# ======================================================
epochs, loss, psnr, ssim = [], [], [], []

with open(os.path.join(LOGS, "result_log2.csv")) as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        epochs.append(int(r[0]))
        loss.append(float(r[1]))
        psnr.append(float(r[2]))
        ssim.append(float(r[3]))

def save_curve(x, y, title, ylabel, name):
    plt.figure()
    plt.plot(x, y, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(VIS, name), dpi=300)
    plt.close()

save_curve(epochs, loss, "Training Loss Curve", "Loss", "training_loss_curve.jpg")
save_curve(epochs, psnr, "PSNR vs Epoch", "PSNR (dB)", "psnr_curve.jpg")
save_curve(epochs, ssim, "SSIM vs Epoch", "SSIM", "ssim_curve.jpg")

# ======================================================
# 2️⃣ LOAD MODEL & DATA
# ======================================================
val_files = [os.path.join(DATA,f) for f in os.listdir(DATA) if f.endswith(".h5")]
dataset = FastMRISliceDataset(val_files)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResidualAttentionUNet().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# ======================================================
# 3️⃣ INPUT | OUTPUT | GT GRID
# ======================================================
fig, axs = plt.subplots(3, 5, figsize=(15, 8))
for i in range(5):
    x, y = dataset[i]
    with torch.no_grad():
        p = model(x.unsqueeze(0).to(device))[0,0].cpu()

    axs[0,i].imshow(x[0], cmap="gray"); axs[0,i].set_title("Input")
    axs[1,i].imshow(p, cmap="gray"); axs[1,i].set_title("Reconstruction")
    axs[2,i].imshow(y[0], cmap="gray"); axs[2,i].set_title("Ground Truth")

    for r in range(3):
        axs[r,i].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(VIS, "compare_grid.jpg"), dpi=300)
plt.close()

# ======================================================
# 4️⃣ ERROR MAP (ABS DIFFERENCE)
# ======================================================
error = torch.abs(p - y[0])

plt.figure(figsize=(4,4))
plt.imshow(error, cmap="hot")
plt.colorbar()
plt.title("Error Map |GT − Prediction|")
plt.axis("off")
plt.savefig(os.path.join(VIS, "error_map.jpg"), dpi=300)
plt.close()

# ======================================================
# 5️⃣ INTENSITY HISTOGRAM
# ======================================================
plt.figure()
plt.hist(y[0].numpy().flatten(), bins=100, alpha=0.6, label="GT")
plt.hist(p.numpy().flatten(), bins=100, alpha=0.6, label="Reconstruction")
plt.legend()
plt.title("Intensity Distribution")
plt.savefig(os.path.join(VIS, "histogram.jpg"), dpi=300)
plt.close()

# ======================================================
# 6️⃣ INDIVIDUAL RECONSTRUCTION SAMPLES
# ======================================================
for i in range(3):
    x, y = dataset[i]
    with torch.no_grad():
        p = model(x.unsqueeze(0).to(device))[0,0].cpu()

    plt.figure(figsize=(4,4))
    plt.imshow(p, cmap="gray")
    plt.axis("off")
    plt.title(f"Reconstruction Sample {i+1}")
    plt.savefig(os.path.join(VIS, f"sample_recon_{i+1:02d}.jpg"), dpi=300)
    plt.close()

print("✅ All visualizations saved in final_result/visualize/")
