#!/bin/bash
# Swin V2 Anti-Overfitting Training Script
# 从头训练，解决过拟合问题

set -e  # Exit on error

# 实验描述（可选参数，用于区分不同实验）
EXP_DESC="${1:-default}"  # 默认为 "default"，可通过 ./train_anti_overfit.sh exp_name 指定

echo "========================================================================"
echo "🚀 Swin V2 Anti-Overfitting Training"
echo "========================================================================"
echo ""
echo "实验名称: $EXP_DESC"
echo ""
echo "配置摘要:"
echo "  - 模型: Swin V2 Base (87M params)"
echo "  - 等效 Batch Size: 128 (物理=32, accumulation=4)"
echo "  - Weight Decay: 0.15 (强正则化)"
echo "  - Label Smoothing: 0.2"
echo "  - Drop Path Rate: 0.35"
echo "  - 数据增强: Medium"
echo "  - 训练轮数: 200 epochs (early stop patience=15)"
echo ""
echo "预期结果:"
echo "  - Val Acc: 0.975+ (当前 0.9676)"
echo "  - Train/Val Gap: < 0.01 (当前 ~0.023)"
echo ""
echo "========================================================================"
echo ""

# 检查必要文件
echo "🔍 检查数据集..."
if [ ! -f "./datasets/train.csv" ]; then
    echo "❌ 错误: datasets/train.csv 不存在"
    exit 1
fi

if [ ! -d "./datasets/train" ]; then
    echo "❌ 错误: datasets/train 目录不存在"
    exit 1
fi

echo "✅ 数据集检查通过"
echo ""

# 创建输出目录（使用 autodl-tmp 大空间 + 实验名称区分）
echo "📁 创建输出目录..."
CHECKPOINT_DIR="/root/autodl-tmp/checkpoints_swin_anti_overfit/${EXP_DESC}"
LOG_DIR="/root/tf-logs/swin_anti_overfit/${EXP_DESC}"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p results/best_models_backup  # 备份目录
echo "✅ 目录创建完成"
echo "   Checkpoints: $CHECKPOINT_DIR"
echo "   Logs: $LOG_DIR"
echo ""

# 启动 TensorBoard (后台) - 指向父目录，自动展示所有实验
echo "📊 启动 TensorBoard..."
kill $(lsof -t -i:6006) 2>/dev/null || true  # 关闭已有 TensorBoard
tensorboard --logdir /root/tf-logs/swin_anti_overfit --port 6006 --bind_all > /dev/null 2>&1 &
TB_PID=$!
echo "✅ TensorBoard 已启动 (PID: $TB_PID)"
echo "   访问: http://localhost:6006"
echo "   当前实验: $EXP_DESC (会自动出现在左侧列表)"
echo ""

# 开始训练（传递目录参数到 Hydra）
echo "🎯 开始训练..."
echo "========================================================================"
echo ""

python train.py -cn swin_v2_anti_overfit \
  checkpoint_dir="$CHECKPOINT_DIR" \
  log_dir="$LOG_DIR"

# 训练完成 - 自动备份 best_model
echo ""
echo "========================================================================"
echo "✅ 训练完成！开始备份..."
echo ""

# 生成时间戳目录名（包含实验描述）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="results/best_models_backup/swin_anti_overfit_${EXP_DESC}_${TIMESTAMP}"

# 创建备份目录并复制 best_model
if [ -f "$CHECKPOINT_DIR/best_model.pt" ]; then
    mkdir -p "$BACKUP_DIR"
    cp "$CHECKPOINT_DIR/best_model.pt" "$BACKUP_DIR/"
    echo "✅ Best model 已备份到: $BACKUP_DIR/best_model.pt"
    
    # 提取并保存训练信息
    python << EOF
import torch
try:
    ckpt = torch.load('$BACKUP_DIR/best_model.pt', map_location='cpu', weights_only=False)
    with open('$BACKUP_DIR/info.txt', 'w') as f:
        f.write(f"实验名称: ${EXP_DESC}\n")
        f.write(f"训练时间: ${TIMESTAMP}\n")
        f.write(f"Epoch: {ckpt.get('epoch', 'N/A')}\n")
        f.write(f"Best Val Acc: {ckpt.get('best_val_acc', 0):.4f}%\n")
        f.write(f"\n配置摘要:\n")
        f.write(f"  - Batch Size: {ckpt.get('config', {}).get('dataset', {}).get('batch_size', 'N/A')}\n")
        f.write(f"  - LR: {ckpt.get('config', {}).get('training', {}).get('optimizer', {}).get('lr', 'N/A')}\n")
        f.write(f"  - Weight Decay: {ckpt.get('config', {}).get('training', {}).get('optimizer', {}).get('weight_decay', 'N/A')}\n")
        f.write(f"  - Label Smoothing: {ckpt.get('config', {}).get('training', {}).get('label_smoothing', 'N/A')}\n")
    print("✅ 训练信息已保存")
except Exception as e:
    print(f"⚠️  保存信息失败: {e}")
EOF
else
    echo "⚠️  未找到 best_model.pt，跳过备份"
fi

echo ""
echo "检查点位置:"
echo "  - 训练目录: $CHECKPOINT_DIR"
echo "  - 备份位置: $BACKUP_DIR/"
echo "  - Logs: $LOG_DIR"
echo ""
echo "下一步:"
echo "  1. 查看 TensorBoard: http://localhost:6006"
echo "  2. 检查备份目录中的 info.txt"
echo "  3. 运行 inference.py 使用备份的模型"
echo ""
echo "💡 提示: autodl-tmp 中的文件可以定期清理，备份在 results/ 下永久保存"
echo "========================================================================"
