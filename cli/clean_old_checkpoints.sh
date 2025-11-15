#!/bin/bash
# 清理旧的完整checkpoint,只保留inference版本
# 谨慎使用!删除前会显示要删除的文件

echo "将要删除以下完整checkpoint文件:"
echo "========================================"
ls -lh results/checkpoints/*.pt | grep -v "_inference.pt" | grep -v "_weights_only.pt" | awk '{printf "%s  %s\n", $5, $9}'
echo "========================================"
echo ""
echo "保留以下轻量级文件:"
ls -lh results/checkpoints/*_inference.pt | awk '{printf "%s  %s\n", $5, $9}'
echo ""

read -p "确认删除? (yes/no): " confirm

if [ "$confirm" == "yes" ]; then
    echo "删除中..."
    rm -v results/checkpoints/checkpoint_epoch_*.pt results/checkpoints/best_model.pt
    echo ""
    echo "✓ 完成!可节省约 6GB 空间"
    echo ""
    echo "剩余文件:"
    ls -lh results/checkpoints/*.pt
else
    echo "取消删除"
fi
