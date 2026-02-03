import csv

CSV_PATH = "final_result/logs/result_log2.csv"
TARGET_EPOCH = 50

with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row["Epoch"]) == TARGET_EPOCH:
            print("✅ Metrics for Epoch 50")
            print(f"Loss : {row['TrainLoss'] if 'TrainLoss' in row else row['Loss']}")
            print(f"PSNR : {row['ValPSNR'] if 'ValPSNR' in row else row['PSNR']} dB")
            print(f"SSIM : {row['ValSSIM'] if 'ValSSIM' in row else row['SSIM']}")
            break
