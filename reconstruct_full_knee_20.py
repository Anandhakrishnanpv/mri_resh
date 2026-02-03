import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import FastMRISliceDataset
from model import ResidualAttentionUNet

# ================= PATHS =================
ROOT = os.path.expanduser("~/Anandhakrishnan P V")
DATA = os.path.join(ROOT, "data", "val")
OUT  = os.path.join(ROOT, "final_research_values", "full_knee_20")

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

# ================= SELECT ONLY 20 SLICES =================
NUM_SLICES = 20
num_available = len(dataset)

if num_available < NUM_SLICES:
    print(f"⚠️ Only {num_available} slices available, using all")
    indices = list(range(num_available))
else:
    # take center 20 slices (best anatomical continuity)
    center = num_available // 2
    start = center - NUM_SLICES // 2
    indices = list(range(start, start + NUM_SLICES))

recon_slices = []
gt_slices = []

with torch.no_grad():
    for idx in indices:
        x, y = dataset[idx]
        pred = model(x.unsqueeze(0).to(device))[0, 0].cpu()

        recon_slices.append(pred.numpy())
        gt_slices.append(y[0].numpy())

# ================= STACK =================
recon_stack = np.stack(recon_slices, axis=0)
gt_stack    = np.stack(gt_slices, axis=0)

# ================= NORMALIZE =================
def normalize(vol):
    return (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

recon_stack = normalize(recon_stack)
gt_stack    = normalize(gt_stack)

# ================= MOSAIC (5 x 4 = 20 slices) =================
def make_mosaic(volume, cols=5):
    slices, H, W = volume.shape
    rows = int(np.ceil(slices / cols))

    canvas = np.zeros((rows * H, cols * W))
    for i in range(slices):
        r = i // cols
        c = i % cols
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = volume[i]

    return canvas

recon_mosaic = make_mosaic(recon_stack, cols=5)
gt_mosaic    = make_mosaic(gt_stack, cols=5)

# ================= SAVE IMAGES =================
plt.figure(figsize=(10, 8))
plt.imshow(recon_mosaic, cmap="gray")
plt.title("Full Knee Reconstruction (20 Slices)")
plt.axis("off")
plt.savefig(os.path.join(OUT, "full_knee_reconstruction_20.jpg"), dpi=300)
plt.close()

plt.figure(figsize=(10, 8))
plt.imshow(gt_mosaic, cmap="gray")
plt.title("Full Knee Ground Truth (20 Slices)")
plt.axis("off")
plt.savefig(os.path.join(OUT, "full_knee_ground_truth_20.jpg"), dpi=300)
plt.close()

print("✅ Full knee (20 slices) reconstruction saved")
print(f"📁 Location: {OUT}")
