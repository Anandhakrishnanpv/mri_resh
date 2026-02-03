import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import FastMRISliceDataset
from model import ResidualAttentionUNet
from utils import compute_psnr

# ================= PATHS =================
ROOT = os.path.expanduser("~/Anandhakrishnan P V")
DATA = os.path.join(ROOT, "data", "val")

# 🔴 REQUIRED FOLDER NAME (AS YOU ASKED)
OUT_DIR = os.path.join(
    ROOT,
    "final_research_values",
    "all_image_rcon"
)

CKPT = os.path.join(
    ROOT,
    "final_result",
    "checkpoints",
    "best_checkpoint.pth"
)

os.makedirs(OUT_DIR, exist_ok=True)

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

# ================= SETTINGS =================
NUM_IMAGES = 100
num_available = len(dataset)
indices = list(range(min(NUM_IMAGES, num_available)))

# ================= NORMALIZATION (CLEAR IMAGES) =================
def normalize(img, pmin=1, pmax=99):
    img = img.numpy()
    lo, hi = np.percentile(img, (pmin, pmax))
    img = np.clip(img, lo, hi)
    return (img - lo) / (hi - lo + 1e-8)

# ================= GENERATE IMAGES =================
for i, idx in enumerate(indices, 1):
    x, y = dataset[idx]

    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device))[0, 0].cpu()

    # -------- PSNR --------
    psnr_input = compute_psnr(x[0], y[0]).item()
    psnr_pred  = compute_psnr(pred, y[0]).item()

    xin  = normalize(x[0])
    xout = normalize(pred)
    xgt  = normalize(y[0])

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(xin, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title(f"Input (Zero-filled)\nPSNR = {psnr_input:.2f} dB")

    axs[1].imshow(xout, cmap="gray", vmin=0, vmax=1)
    axs[1].set_title(f"Reconstruction (Proposed)\nPSNR = {psnr_pred:.2f} dB")

    axs[2].imshow(xgt, cmap="gray", vmin=0, vmax=1)
    axs[2].set_title("Ground Truth")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, f"recon_{i:03d}.jpg"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

print(f"✅ {len(indices)} clear MRI reconstructions saved")
print(f"📁 Location: {OUT_DIR}")
