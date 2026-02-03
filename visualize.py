import os, torch, random
import matplotlib.pyplot as plt
from dataset import FastMRISliceDataset
from model import ResidualAttentionUNet

ROOT = os.path.expanduser("~/Anandhakrishnan P V")
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "final_result/img_final")
os.makedirs(OUT, exist_ok=True)

val_files = [os.path.join(DATA, "val", f) for f in os.listdir(os.path.join(DATA, "val")) if f.endswith(".h5")]
dataset = FastMRISliceDataset(val_files)

model = ResidualAttentionUNet()
model.load_state_dict(torch.load(f"{ROOT}/final_result/checkpoints/best_checkpoint.pth", map_location="cpu"))
model.eval()

indices = random.sample(range(len(dataset)), 5)

for idx in indices:
    x, y = dataset[idx]
    with torch.no_grad():
        pred = model(x.unsqueeze(0))[0,0]

    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    axs[0].imshow(x[0], cmap="gray"); axs[0].set_title("Input")
    axs[1].imshow(pred, cmap="gray"); axs[1].set_title("Reconstruction")
    axs[2].imshow(y[0], cmap="gray"); axs[2].set_title("Ground Truth")

    for ax in axs: ax.axis("off")
    plt.savefig(f"{OUT}/sample_{idx}.jpg", dpi=300)
    plt.close()

print("✅ Visualization saved")
