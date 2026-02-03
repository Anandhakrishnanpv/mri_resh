import csv

CSV = "final_result/logs/result_log2.csv"

print(f"{'Epoch':<6} {'Loss':<10} {'PSNR(dB)':<10} {'SSIM':<10}")
print("-"*40)

with open(CSV) as f:
    reader = csv.DictReader(f)
    for r in reader:
        print(f"{r['Epoch']:<6} {float(r['Loss']):<10.4f} {float(r['PSNR']):<10.2f} {float(r['SSIM']):<10.4f}")
