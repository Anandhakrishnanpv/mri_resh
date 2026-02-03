import os
import torch
import random
import matplotlib.pyplot as plt

from dataset import FastMRISliceDataset
from model import ResidualAttentionUNet
from utils import compute_psnr

# ================= PATHS =================
ROOT = os.path.expanduser("~/Anandhakrishnan P V")
DATA = os.path.join(ROOT, "data", "val")
OUT  = os.path.join(ROOT, "final_result", "img_final")
CKPT = os.path.join(ROOT, "final_result", "checkpoints", "last_checkpoint.pth")

os.makedirs(OUT, exist_ok=True)

# ================= LOAD DATA =================
val_files = [
    os.path.join(DATA, f)
    for f in os.listdir(DATA)
    if f.endswith(".h5")
]

dataset = FastMRISliceDataset(val_files)

num_samples = min(10, len(dataset))
indices = random.sample(range(len(dataset)), num_samples)

# ================= LOAD CHECKPOINT (WITH EPOCH) =================
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(CKPT, map_location=device)

epoch_trained = checkpoint["epoch"]  # ✅ ACTUAL EPOCH NUMBER

model = ResidualAttentionUNet().to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

# ================= RECONSTRUCTION =================
for i, idx in enumerate(indices, 1):
    x, y = dataset[idx]

    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device))[0, 0].cpu()

    # -------- METRICS --------
    psnr_input = compute_psnr(x[0], y[0]).item()
    psnr_pred  = compute_psnr(pred, y[0]).item()

    # -------- VISUALIZATION --------
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(x[0], cmap="gray")
    axs[0].set_title(
        f"Input (Zero-filled)\nPSNR: {psnr_input:.2f} dB"
    )

    axs[1].imshow(pred, cmap="gray")
    axs[1].set_title(
        f"Reconstruction (Epoch {epoch_trained})\nPSNR: {psnr_pred:.2f} dB"
    )

    axs[2].imshow(y[0], cmap="gray")
    axs[2].set_title("Ground Truth")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUT, f"compare_epoch_{epoch_trained:03d}_img_{i:02d}.jpg"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

print(f"✅ {num_samples} epoch-accurate comparison images saved")
