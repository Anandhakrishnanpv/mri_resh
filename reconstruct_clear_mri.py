import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from dataset import FastMRISliceDataset
from model import ResidualAttentionUNet
from utils import compute_psnr

# ================= PATHS =================
ROOT = os.path.expanduser("~/Anandhakrishnan P V")
DATA = os.path.join(ROOT, "data", "val")

OUT_DIR = os.path.join(
    ROOT,
    "final_research_values",
    "clear_reconstructions_100"
)

CKPT = os.path.join(
    ROOT,
    "final_result",
    "checkpoints",
    "best_checkpoint.pth"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ================= LOAD DATA =================
val_files = [
    os.path.join(DATA, f)
    for f in os.listdir(DATA)
    if f.endswith(".h5")
]

dataset = FastMRISliceDataset(val_files)

# ================= LOAD MODEL =================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResidualAttentionUNet().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# ================= SETTINGS =================
NUM_IMAGES = 100
num_available = len(dataset)

if num_available < NUM_IMAGES:
    print(f"⚠️ Only {num_available} samples available, generating all of them")
    indices = list(range(num_available))
else:
    indices = random.sample(range(num_available), NUM_IMAGES)

# ================= NORMALIZATION =================
def normalize(img):
    img = img.numpy()
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

# ================= GENERATE IMAGES =================
for i, idx in enumerate(indices, 1):
    x, y = dataset[idx]

    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device))[0, 0].cpu()

    psnr_input = compute_psnr(x[0], y[0]).item()
    psnr_pred  = compute_psnr(pred, y[0]).item()

    xin  = normalize(x[0])
    xout = normalize(pred)
    xgt  = normalize(y[0])

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(xin, cmap="gray")
    axs[0].set_title(f"Input (Zero-filled)\nPSNR: {psnr_input:.2f} dB")

    axs[1].imshow(xout, cmap="gray")
    axs[1].set_title(f"Reconstruction\nPSNR: {psnr_pred:.2f} dB")

    axs[2].imshow(xgt, cmap="gray")
    axs[2].set_title("Ground Truth")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, f"clear_recon_{i:03d}.jpg"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

print(f"✅ {len(indices)} clear MRI reconstruction images saved")
print(f"📁 Location: {OUT_DIR}")
