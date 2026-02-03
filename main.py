import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_msssim

from utils import set_seed, compute_psnr
from dataset import FastMRISliceDataset
from model import ResidualAttentionUNet

set_seed(42)

ROOT = os.path.expanduser("~/Anandhakrishnan P V")
DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(ROOT, "checkpoints")
os.makedirs(CKPT, exist_ok=True)

train_files = [os.path.join(DATA, "train", f)
               for f in os.listdir(os.path.join(DATA, "train")) if f.endswith(".h5")]
val_files = [os.path.join(DATA, "val", f)
             for f in os.listdir(os.path.join(DATA, "val")) if f.endswith(".h5")]

train_ds = FastMRISliceDataset(train_files)
val_ds   = FastMRISliceDataset(val_files)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResidualAttentionUNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
best_psnr = 0

for epoch in range(40):
    model.train()
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = F.l1_loss(pred, y) + (1 - pytorch_msssim.ssim(pred, y, data_range=1.0))
        loss.backward()
        optimizer.step()

    model.eval()
    psnr = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            psnr += compute_psnr(model(x), y).item()

    psnr /= len(val_loader)
    print(f"Epoch {epoch+1} | PSNR: {psnr:.2f}")

    if psnr > best_psnr:
        best_psnr = psnr
        torch.save(model.state_dict(), f"{CKPT}/best_model.pth")
        print("💾 Saved best model")
