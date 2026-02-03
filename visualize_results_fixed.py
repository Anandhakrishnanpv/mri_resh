import os
import csv
import torch
import matplotlib.pyplot as plt

from dataset import FastMRISliceDataset
from model import ResidualAttentionUNet

# ================= PATHS =================
ROOT = os.path.expanduser("~/Anandhakrishnan P V")
FINAL_ROOT = os.path.join(ROOT, "img_result")
LOGS = os.path.join(FINAL_ROOT, "logs")

VIS_BASE = os.path.join(FINAL_ROOT, "visualize")
VIS = os.path.join(VIS_BASE, "final")   # ✅ REQUIRED FOLDER

DATA = os.path.join(ROOT, "data", "val")
CKPT = os.path.join(FINAL_ROOT, "checkpoints", "last_checkpoint.pth")

os.makedirs(VIS, exist_ok=True)

# ======================================================
# 1️⃣ TRAINING CURVES (LOSS / PSNR / SSIM)
# ======================================================
epochs, loss, psnr, ssim = [], [], [], []

csv_path = os.path.join(LOGS, "result_log2.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError("❌ result_log2.csv not found")

with open(csv_path) as f:
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
    plt.tight_layout()
    plt.savefig(os.path.join(VIS, name), dpi=300)
    plt.close()

save_curve(epochs, loss, "Training Loss Curve", "Loss", "training_loss_curve.jpg")
save_curve(epochs, psnr, "PSNR vs Epoch", "PSNR (dB)", "psnr_curve.jpg")
save_curve(epochs, ssim, "SSIM vs Epoch", "SSIM", "ssim_curve.jpg")

# ======================================================
# 2️⃣ LOAD MODEL & DATA (EPOCH-AWARE)
# ======================================================
val_files = [
    os.path.join(DATA, f)
    for f in os.listdir(DATA)
    if f.endswith(".h5")
]

dataset = FastMRISliceDataset(val_files)
num_samples = min(5, len(dataset))

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(CKPT, map_location=device)
epoch_trained = checkpoint["epoch"]

model = ResidualAttentionUNet().to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

# ======================================================
# 3️⃣ INPUT | RECONSTRUCTION | GT GRID
# ======================================================
fig, axs = plt.subplots(3, num_samples, figsize=(3*num_samples, 8))
stored = []

for i in range(num_samples):
    x, y = dataset[i]
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device))[0, 0].cpu()

    stored.append((x[0], pred, y[0]))

    axs[0, i].imshow(x[0], cmap="gray")
    axs[0, i].set_title("Input")

    axs[1, i].imshow(pred, cmap="gray")
    axs[1, i].set_title(f"Reconstruction\n(Epoch {epoch_trained})")

    axs[2, i].imshow(y[0], cmap="gray")
    axs[2, i].set_title("Ground Truth")

    for r in range(3):
        axs[r, i].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(VIS, "compare_grid.jpg"), dpi=300)
plt.close()

# ======================================================
# 4️⃣ ERROR MAP (CONSISTENT SAMPLE)
# ======================================================
x0, p0, y0 = stored[0]
error = torch.abs(p0 - y0)

plt.figure(figsize=(4, 4))
plt.imshow(error, cmap="hot")
plt.colorbar()
plt.title("|GT − Reconstruction|")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(VIS, "error_map.jpg"), dpi=300)
plt.close()

# ======================================================
# 5️⃣ INTENSITY HISTOGRAM
# ======================================================
plt.figure()
plt.hist(y0.numpy().flatten(), bins=100, alpha=0.6, label="Ground Truth")
plt.hist(p0.numpy().flatten(), bins=100, alpha=0.6, label="Reconstruction")
plt.legend()
plt.title("Intensity Distribution")
plt.tight_layout()
plt.savefig(os.path.join(VIS, "histogram.jpg"), dpi=300)
plt.close()

# ======================================================
# 6️⃣ INDIVIDUAL RECONSTRUCTION SAMPLES
# ======================================================
for i, (_, p, _) in enumerate(stored[:3], 1):
    plt.figure(figsize=(4, 4))
    plt.imshow(p, cmap="gray")
    plt.axis("off")
    plt.title(f"Reconstruction Sample {i} (Epoch {epoch_trained})")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS, f"sample_recon_{i:02d}.jpg"), dpi=300)
    plt.close()

print("✅ All visualization images saved in final_result/visualize/final/")
