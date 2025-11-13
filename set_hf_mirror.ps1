# 设置 HuggingFace 镜像 - 解决预训练权重下载问题
# 运行方式: . .\set_hf_mirror.ps1

Write-Host "Setting HuggingFace Mirror..." -ForegroundColor Green

# 设置当前会话的环境变量
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = ".\cache\huggingface"
$env:TORCH_HOME = ".\cache\torch"

Write-Host "✓ HF_ENDPOINT = $env:HF_ENDPOINT" -ForegroundColor Cyan
Write-Host "✓ HF_HOME = $env:HF_HOME" -ForegroundColor Cyan  
Write-Host "✓ TORCH_HOME = $env:TORCH_HOME" -ForegroundColor Cyan

Write-Host "`nMirror configured successfully!" -ForegroundColor Green
Write-Host "Now you can run: python train.py ..." -ForegroundColor Yellow
