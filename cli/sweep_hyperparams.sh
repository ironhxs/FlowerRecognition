#!/bin/bash
# Hydra 超参数扫描脚本
# 自动运行多个实验，寻找最佳超参数组合

set -e

echo "========================================================================"
echo "🔍 Swin V2 超参数扫描"
echo "========================================================================"
echo ""
echo "扫描范围:"
echo "  - Learning Rate: 1.0e-5, 1.5e-5, 2.0e-5, 2.5e-5"
echo "  - Weight Decay: 0.12, 0.15, 0.18"
echo "  - Label Smoothing: 0.15, 0.2"
echo "  - Drop Path Rate: 0.3, 0.35"
echo ""
echo "总实验数: 4 × 3 × 2 × 2 = 48 组合"
echo "每组 30 epochs，预计总时间: ~40 小时"
echo ""
read -p "确认开始扫描? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi
echo "========================================================================"
echo ""

# 创建输出目录
mkdir -p /root/autodl-tmp/checkpoints_sweep
mkdir -p /root/tf-logs/sweep

# 启动 TensorBoard
echo "📊 启动 TensorBoard..."
kill $(lsof -t -i:6006) 2>/dev/null || true
tensorboard --logdir /root/tf-logs/sweep --port 6006 --bind_all > /dev/null 2>&1 &
echo "✅ TensorBoard: http://localhost:6006"
echo ""

# 开始扫描
echo "🚀 开始超参数扫描..."
echo "========================================================================"
echo ""

python train.py -cn sweep_lr --multirun \
  hydra.sweep.dir=/root/autodl-tmp/sweep_outputs \
  hydra.sweep.subdir='lr=${training.optimizer.lr}_wd=${training.optimizer.weight_decay}_ls=${training.label_smoothing}_dp=${training.drop_path_rate}'

echo ""
echo "========================================================================"
echo "✅ 超参数扫描完成！"
echo ""
echo "查看结果:"
echo "  1. TensorBoard: http://localhost:6006 (对比所有实验曲线)"
echo "  2. 输出目录: /root/autodl-tmp/sweep_outputs/"
echo "  3. Logs: /root/tf-logs/sweep/"
echo ""
echo "下一步: 找出 Val Acc 最高的配置，更新到 swin_anti_overfit.yaml"
echo "========================================================================"
