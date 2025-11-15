#!/bin/bash
# 学习率扫描 - 带 MixUp/CutMix 版本
# 用于对比数据增强的效果

set -e

echo "========================================================================"
echo "🎯 学习率扫描 (MixUp/CutMix 增强版)"
echo "========================================================================"
echo ""
echo "扫描范围: 2.2e-5, 2.6e-5, 3.0e-5, 3.5e-5, 4.0e-5"
echo "固定参数: weight_decay=0.15, label_smoothing=0.2, drop_path_rate=0.35"
echo "新增: MixUp + CutMix (alpha=0.2/1.0, prob=0.5)"
echo "训练轮数: 20 epochs"
echo "总实验数: 5 组 (重点测试大 LR)"
echo "预计时间: ~1.5 小时"
echo ""
echo "💡 对比目的: 验证 MixUp/CutMix 是否能进一步提升 Val Acc"
echo ""
echo "========================================================================"
echo ""

# 创建新的独立目录
mkdir -p /root/autodl-tmp/checkpoints_sweep_lr_mixup
mkdir -p /root/tf-logs/sweep_lr_mixup

# 启动 TensorBoard (指向 MixUp 版本)
echo "📊 启动 TensorBoard..."
kill $(lsof -t -i:6006) 2>/dev/null || true
tensorboard --logdir /root/tf-logs/sweep_lr_mixup --port 6006 --bind_all > /dev/null 2>&1 &
echo "✅ TensorBoard: http://localhost:6006"
echo ""
echo "💡 提示: 对比旧结果可查看 /root/tf-logs/sweep_lr/"
echo ""

# 精选 LR 列表（MixUp 允许更大 LR，扩大搜索范围）
# 理论: MixUp 梯度更平滑 → 可以用更大学习率
LR_VALUES=(2.2e-5 2.6e-5 3.0e-5 3.5e-5 4.0e-5)

# 循环训练
for lr in "${LR_VALUES[@]}"; do
    exp_name="lr_${lr}_mixup"
    echo "========================================================================"
    echo "🚀 实验: $exp_name"
    echo "========================================================================"
    echo ""
    
    python train.py -cn swin_v2_anti_overfit \
      training.optimizer.lr=$lr \
      training.epochs=40 \
      checkpoint_dir="/root/autodl-tmp/checkpoints_sweep_lr_mixup/${exp_name}" \
      log_dir="/root/tf-logs/sweep_lr_mixup/${exp_name}"
    
    echo ""
    echo "✅ $exp_name 完成"
    echo ""
    
    # 清理中间 checkpoint（立即释放空间）
    find "/root/autodl-tmp/checkpoints_sweep_lr_mixup/${exp_name}" -name "checkpoint_epoch_*.pt" -delete
    
    # 提取权重（节省空间）
    if [ -f "/root/autodl-tmp/checkpoints_sweep_lr_mixup/${exp_name}/best_model.pt" ]; then
        python cli/extract_weights.py \
          "/root/autodl-tmp/checkpoints_sweep_lr_mixup/${exp_name}/best_model.pt" \
          --delete-original
    fi
done

echo ""
echo "========================================================================"
echo "✅ MixUp/CutMix 学习率扫描完成！"
echo ""
echo "📊 结果对比:"
echo ""

# 对比新旧结果
for lr in "${LR_VALUES[@]}"; do
    exp_new="lr_${lr}_mixup"
    exp_old="lr_${lr}"
    
    ckpt_new="/root/autodl-tmp/checkpoints_sweep_lr_mixup/${exp_new}/best_model_weights_only.pt"
    ckpt_old="/root/autodl-tmp/checkpoints_sweep_lr/${exp_old}/best_model_weights_only.pt"
    
    if [ -f "$ckpt_new" ]; then
        acc_new=$(python -c "import torch; ckpt=torch.load('$ckpt_new', map_location='cpu', weights_only=False); print(f\"{ckpt.get('best_val_acc', 0):.2f}\")" 2>/dev/null || echo "N/A")
        acc_old=$(python -c "import torch; ckpt=torch.load('$ckpt_old', map_location='cpu', weights_only=False); print(f\"{ckpt.get('best_val_acc', 0):.2f}\")" 2>/dev/null || echo "N/A")
        
        if [ "$acc_new" != "N/A" ] && [ "$acc_old" != "N/A" ]; then
            diff=$(python -c "print(f\"{float('$acc_new') - float('$acc_old'):+.2f}\")")
            echo "  LR=$lr: $acc_old% → $acc_new% ($diff%)"
        else
            echo "  LR=$lr: Old=$acc_old%, New=$acc_new%"
        fi
    fi
done

echo ""
echo "TensorBoard 对比:"
echo "  - 旧版 (无 MixUp): tensorboard --logdir /root/tf-logs/sweep_lr --port 6007"
echo "  - 新版 (MixUp):   http://localhost:6006 (当前运行)"
echo ""
echo "========================================================================"
