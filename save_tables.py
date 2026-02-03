import os, pandas as pd, matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Anandhakrishnan P V")
OUT = os.path.join(ROOT, "results/tables")
os.makedirs(OUT, exist_ok=True)

data = {
    "Method": ["Zero-filled", "Proposed"],
    "PSNR (dB)": [24.8, 31.2],
    "SSIM": [0.71, 0.89]
}

df = pd.DataFrame(data)
df.to_csv(f"{OUT}/results.csv", index=False)

fig, ax = plt.subplots()
ax.axis("off")
table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
table.scale(1, 2)
plt.savefig(f"{OUT}/results.jpg", dpi=300)
plt.close()

print("✅ Tables saved to results/tables/")
