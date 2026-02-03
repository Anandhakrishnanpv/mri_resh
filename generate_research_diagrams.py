import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ================= PATHS =================
ROOT = os.path.expanduser("~/Anandhakrishnan P V")
BASE_DIR = os.path.join(ROOT, "final_research_values")

CSV_FILE = os.path.join(BASE_DIR, "result_log2.csv")
OUT_DIR  = os.path.join(BASE_DIR, "research_diagrams")

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"❌ {CSV_FILE} not found")

os.makedirs(OUT_DIR, exist_ok=True)

print(f"✅ Using metrics file: {CSV_FILE}")

# ================= LOAD CSV =================
epochs, loss, psnr, ssim = [], [], [], []

with open(CSV_FILE) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        epochs.append(int(row[0]))
        loss.append(float(row[1]))
        psnr.append(float(row[2]))
        ssim.append(float(row[3]))

epochs = np.array(epochs)
loss   = np.array(loss)
psnr   = np.array(psnr)
ssim   = np.array(ssim)

# ================= SAVE HELPER =================
def save_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=300)
    plt.close()

# =====================================================
# 1️⃣ Training Loss vs Epoch
# =====================================================
plt.figure()
plt.plot(epochs, loss, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
save_plot("01_loss_vs_epoch.jpg")

# =====================================================
# 2️⃣ PSNR vs Epoch
# =====================================================
plt.figure()
plt.plot(epochs, psnr, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs Epoch")
plt.grid(True)
save_plot("02_psnr_vs_epoch.jpg")

# =====================================================
# 3️⃣ SSIM vs Epoch
# =====================================================
plt.figure()
plt.plot(epochs, ssim, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.title("SSIM vs Epoch")
plt.grid(True)
save_plot("03_ssim_vs_epoch.jpg")

# =====================================================
# 4️⃣ Loss + PSNR (Dual Axis)
# =====================================================
fig, ax1 = plt.subplots()
ax1.plot(epochs, loss)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")

ax2 = ax1.twinx()
ax2.plot(epochs, psnr)
ax2.set_ylabel("PSNR (dB)")

plt.title("Loss and PSNR vs Epoch")
save_plot("04_loss_psnr_dual_axis.jpg")

# =====================================================
# 5️⃣ PSNR Distribution
# =====================================================
plt.figure()
plt.hist(psnr, bins=15)
plt.xlabel("PSNR (dB)")
plt.ylabel("Frequency")
plt.title("PSNR Distribution")
save_plot("05_psnr_histogram.jpg")

# =====================================================
# 6️⃣ SSIM Distribution
# =====================================================
plt.figure()
plt.hist(ssim, bins=15)
plt.xlabel("SSIM")
plt.ylabel("Frequency")
plt.title("SSIM Distribution")
save_plot("06_ssim_histogram.jpg")

# =====================================================
# 7️⃣ PSNR Box Plot
# =====================================================
plt.figure()
plt.boxplot(psnr)
plt.ylabel("PSNR (dB)")
plt.title("PSNR Box Plot")
save_plot("07_psnr_boxplot.jpg")

# =====================================================
# 8️⃣ SSIM Box Plot
# =====================================================
plt.figure()
plt.boxplot(ssim)
plt.ylabel("SSIM")
plt.title("SSIM Box Plot")
save_plot("08_ssim_boxplot.jpg")

# =====================================================
# 9️⃣ PSNR Improvement Curve
# =====================================================
baseline = np.full_like(psnr, psnr[0])

plt.figure()
plt.plot(epochs, baseline, label="Zero-filled Input")
plt.plot(epochs, psnr, label="Proposed Reconstruction")
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.title("PSNR Improvement Over Training")
plt.legend()
plt.grid(True)
save_plot("09_psnr_improvement.jpg")

# =====================================================
# 🔟 Loss Convergence (Log Scale)
# =====================================================
plt.figure()
plt.semilogy(epochs, loss)
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.title("Loss Convergence")
plt.grid(True)
save_plot("10_loss_convergence_log.jpg")

print("✅ All 10 research diagrams generated successfully")
print(f"📁 Saved in: {OUT_DIR}")
