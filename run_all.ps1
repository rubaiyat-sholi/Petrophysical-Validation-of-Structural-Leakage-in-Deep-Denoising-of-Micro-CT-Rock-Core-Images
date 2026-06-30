# run_all.ps1 — train all three models sequentially, then evaluate
# Usage: powershell -File run_all.ps1

$ErrorActionPreference = "Stop"
$logdir = "runs"

Write-Host "=== Training DnCNN (50 epochs) ===" -ForegroundColor Cyan
py -3 train.py --model dncnn --epochs 50 --batch 32 --workers 0
if (!$?) { Write-Host "DnCNN training failed"; exit 1 }

Write-Host "`n=== Training UNet (50 epochs) ===" -ForegroundColor Cyan
py -3 train.py --model unet --epochs 50 --batch 32 --workers 0
if (!$?) { Write-Host "UNet training failed"; exit 1 }

Write-Host "`n=== Training UNet+Leakage (50 epochs, beta=0.1) ===" -ForegroundColor Cyan
py -3 train.py --model unet_leakage --epochs 50 --batch 32 --workers 0 --beta 0.1
if (!$?) { Write-Host "UNet+Leakage training failed"; exit 1 }

Write-Host "`n=== All models trained. Running evaluation ===" -ForegroundColor Green

$dncnn_ckpt   = (Get-ChildItem "runs\dncnn_*\best.pt"        | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$unet_ckpt    = (Get-ChildItem "runs\unet_2*\best.pt"        | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$leakage_ckpt = (Get-ChildItem "runs\unet_leakage_*\best.pt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName

Write-Host "DnCNN ckpt   : $dncnn_ckpt"
Write-Host "UNet ckpt    : $unet_ckpt"
Write-Host "Leakage ckpt : $leakage_ckpt"

py -3 evaluate.py --dncnn "$dncnn_ckpt" --unet "$unet_ckpt" --unet_leakage "$leakage_ckpt"

Write-Host "`n=== Done. Results in results/ ===" -ForegroundColor Green
