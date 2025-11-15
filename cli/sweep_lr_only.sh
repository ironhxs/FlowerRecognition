#!/bin/bash
# 简化版：只扫描学习率（快速实验）
# 用于快速找到最佳学习率范围

set -e


# 创建目录
mkdir -p /root/autodl-tmp/checkpoints_sweep_lr
mkdir -p /root/tf-logs/sweep_lr

# 启动 TensorBoard
echo "📊 启动 TensorBoard..."
kill $(lsof -t -i:6006) 2>/dev/null || true
tensorboard --logdir /root/tf-logs/sweep_lr --port 6006 --bind_all > /dev/null 2>&1 &
echo "✅ TensorBoard: http://localhost:6006"
echo ""

# 学习率列表（从 8e-6 开始，11 个实验）
LR_VALUES=(8.0e-6)

# 循环训练
for lr in "${LR_VALUES[@]}"; do
    exp_name="lr_${lr}"
    echo "========================================================================"
    echo "🚀 实验: $exp_name"
    echo "========================================================================"
    echo ""
    
    python train.py -cn swin_v2_anti_overfit \
      training.optimizer.lr=$lr \
      training.epochs=20 \
      dataset.val_split=0.05 \
      checkpoint_dir="/root/autodl-tmp/checkpoints_sweep_lr/${exp_name}" \
      log_dir="/root/tf-logs/sweep_lr/${exp_name}"
    
    echo ""
    echo "✅ $exp_name 完成"
    echo ""
    
    # 清理中间 checkpoint（立即释放空间）
    find "/root/autodl-tmp/checkpoints_sweep_lr/${exp_name}" -name "checkpoint_epoch_*.pt" -delete
    
    # 提取权重（节省空间）
    if [ -f "/root/autodl-tmp/checkpoints_sweep_lr/${exp_name}/best_model.pt" ]; then
        python cli/extract_weights.py \
          "/root/autodl-tmp/checkpoints_sweep_lr/${exp_name}/best_model.pt" \
          --delete-original
    fi
done

echo ""
echo "========================================================================"
echo "✅ 学习率扫描完成！"
echo ""
echo "TensorBoard: http://localhost:6006"
echo "  - 左侧会显示 12 条曲线: lr_6.0e-6, lr_8.0e-6, ..., lr_3.0e-5"
echo "  - 对比 Val Acc 找出最佳学习率"
echo ""
echo "查看结果摘要:"
echo ""
for lr in 6.0e-6 8.0e-6 1.0e-5 1.2e-5 1.4e-5 1.6e-5 1.8e-5 2.0e-5 2.2e-5 2.4e-5 2.6e-5 3.0e-5; do
    exp="lr_${lr}"
    ckpt_dir="/root/autodl-tmp/checkpoints_sweep_lr/${exp}"
    if [ -f "${ckpt_dir}/best_model_weights_only.pt" ]; then
        echo "  $exp: $(python -c "import torch; ckpt=torch.load('${ckpt_dir}/best_model_weights_only.pt', weights_only=False); print(f\"Val Acc: {ckpt.get('best_val_acc', 0):.2f}% (Epoch {ckpt.get('epoch', 'N/A')})\")" 2>/dev/null || echo "N/A")"
    fi
done
echo ""
echo "========================================================================"
