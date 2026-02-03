import torch, os, matplotlib.pyplot as plt
from dataset import FastMRISliceDataset
from model import ResidualAttentionUNet

ROOT = os.path.expanduser("~/Anandhakrishnan P V")
DATA = os.path.join(ROOT,"data/val")
IMG  = os.path.join(ROOT,"final_result/img_final")

files = [os.path.join(DATA,f) for f in os.listdir(DATA) if f.endswith(".h5")]
ds = FastMRISliceDataset(files)

model = ResidualAttentionUNet()
model.load_state_dict(torch.load(f"{ROOT}/final_result/checkpoints/best_checkpoint.pth", map_location="cpu"))
model.eval()

x,y = ds[0]
with torch.no_grad():
    p = model(x.unsqueeze(0))[0,0]

plt.imsave(f"{IMG}/final_reconstruction.jpg", p, cmap="gray")
print("✅ Final reconstructed image saved")
