import os
import csv
import sys
import torch
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_msssim

from dataset import FastMRISliceDataset
from model import ResidualAttentionUNet
from utils import set_seed, compute_psnr

# ================= CONFIG =================
EPOCHS = 40
BATCH_SIZE = 8
LR = 2e-4
ACCEL = 4
# ========================================

set_seed(42)

# ================= PATHS =================
ROOT = os.path.expanduser("~/Anandhakrishnan P V")
OUT_DIR = os.path.join(ROOT, "final_research_values")
os.makedirs(OUT_DIR, exist_ok=True)

LIVE_LOG   = os.path.join(OUT_DIR, "live_progress.log")
EPOCH_LOG  = os.path.join(OUT_DIR, "epoch_summary.log")
CSV_FILE   = os.path.join(OUT_DIR, "metrics.csv")
FINAL_FILE = os.path.join(OUT_DIR, "final_values.txt")

# ================= LOGGER (EPOCH SUMMARY) =================
epoch_logger = logging.getLogger("epoch_logger")
epoch_logger.setLevel(logging.INFO)
epoch_handler = logging.FileHandler(EPOCH_LOG)
epoch_handler.setFormatter(logging.Formatter("%(message)s"))
epoch_logger.addHandler(epoch_handler)

# ================= CSV INIT =================
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Epoch", "Loss", "PSNR", "SSIM"])

# ================= DATA =================
DATA = os.path.join(ROOT, "data")

train_files = [
    os.path.join(DATA, "train", f)
    for f in os.listdir(os.path.join(DATA, "train"))
    if f.endswith(".h5")
]

val_files = [
    os.path.join(DATA, "val", f)
    for f in os.listdir(os.path.join(DATA, "val"))
    if f.endswith(".h5")
]

train_ds = FastMRISliceDataset(train_files, accel=ACCEL)
val_ds   = FastMRISliceDataset(val_files, accel=ACCEL)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=1)

# ================= MODEL =================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResidualAttentionUNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ================= LIVE TQDM LOG FILE =================
live_log_fh = open(LIVE_LOG, "a")

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, live_log_fh)
sys.stderr = Tee(sys.stderr, live_log_fh)

# ================= TRAIN =================
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{EPOCHS}",
        file=sys.stdout,
        dynamic_ncols=True
    )

    for x, y in bar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        pred = model(x)
        loss = F.l1_loss(pred, y) + (1 - pytorch_msssim.ssim(pred, y, data_range=1.0))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = total_loss / len(train_loader)

    # ================= VALIDATION =================
    model.eval()
    psnr_sum, ssim_sum = 0.0, 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            psnr_sum += compute_psnr(p, y).item()
            ssim_sum += pytorch_msssim.ssim(p, y, data_range=1.0).item()

    avg_psnr = psnr_sum / len(val_loader)
    avg_ssim = ssim_sum / len(val_loader)

    # ================= SAVE VALUES =================
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow([epoch, train_loss, avg_psnr, avg_ssim])

    epoch_logger.info(
        f"Epoch {epoch}/{EPOCHS}: "
        f"Train Loss={train_loss:.4f} | "
        f"Val PSNR={avg_psnr:.2f} dB | "
        f"Val SSIM={avg_ssim:.4f}"
    )

# ================= FINAL VALUES =================
with open(FINAL_FILE, "w") as f:
    f.write("FINAL TRAINING RESULTS\n")
    f.write("======================\n")
    f.write(f"Epochs : {EPOCHS}\n")
    f.write(f"Loss   : {train_loss:.6f}\n")
    f.write(f"PSNR   : {avg_psnr:.2f} dB\n")
    f.write(f"SSIM   : {avg_ssim:.6f}\n")

print("✅ Training completed successfully")
live_log_fh.close()
