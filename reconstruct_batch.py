import os
import torch
import random
import matplotlib.pyplot as plt

from dataset import FastMRISliceDataset
from model import ResidualAttentionUNet

# -------- PATHS --------
ROOT = os.path.expanduser("~/Anandhakrishnan P V")
DATA = os.path.join(ROOT, "data", "val")
OUT  = os.path.join(ROOT, "final_result", "img_final")
CKPT = os.path.join(ROOT, "final_result", "checkpoints", "best_checkpoint.pth")

os.makedirs(OUT, exist_ok=True)

# -------- LOAD DATA --------
val_files = [
    os.path.join(DATA, f)
    for f in os.listdir(DATA)
    if f.endswith(".h5")
]

dataset = FastMRISliceDataset(val_files)

# -------- LOAD MODEL --------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResidualAttentionUNet().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# -------- SELECT 10 RANDOM SLICES --------
indices = random.sample(range(len(dataset)), 10)

for i, idx in enumerate(indices, 1):
    x, y = dataset[idx]

    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device))[0, 0].cpu()

    # -------- SAVE IMAGE --------
    plt.figure(figsize=(4,4))
    plt.imshow(pred, cmap="gray")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUT, f"reconstructed_{i:02d}.jpg"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()

print("✅ 10 reconstructed MRI knee images saved successfully")
