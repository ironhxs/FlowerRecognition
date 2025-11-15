#!/bin/bash
# 快速导出验证集（自动复制图片）

echo "========================================================================"
echo "📦 导出验证集数据集"
echo "========================================================================"
echo ""

# 运行导出脚本（自动模式）
python << 'EOF'
import os
import sys
from pathlib import Path
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 配置参数（和训练配置保持一致）
TRAIN_CSV = "./datasets/train.csv"
TRAIN_DIR = "./datasets/train"
VAL_SPLIT = 0.15
SEED = 42
OUTPUT_DIR = "./exported_val_dataset"

print("配置参数:")
print(f"  Val Split: {VAL_SPLIT} ({VAL_SPLIT*100:.1f}%)")
print(f"  Random Seed: {SEED}")
print()

# 读取训练数据
df = pd.read_csv(TRAIN_CSV)
print(f"✅ 加载训练数据: {len(df)} 个样本")

# 分层划分
train_ids, val_ids, train_labels, val_labels = train_test_split(
    df['image_id'].tolist(),
    df['label'].tolist(),
    test_size=VAL_SPLIT,
    random_state=SEED,
    stratify=df['label'].tolist()
)

print(f"✅ 划分完成:")
print(f"   Train: {len(train_ids)} 样本")
print(f"   Val:   {len(val_ids)} 样本")
print()

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/val_images", exist_ok=True)

# 保存 val.csv
val_df = pd.DataFrame({
    'image_id': val_ids,
    'label': val_labels
})
val_df = val_df.sort_values('image_id').reset_index(drop=True)
val_df.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
print(f"✅ 保存标签文件: {OUTPUT_DIR}/val.csv")

# 复制图片
print(f"📁 复制验证集图片...")
copied = 0
missing = 0

for image_id in tqdm(val_ids, desc="复制中"):
    src = f"{TRAIN_DIR}/{image_id}"
    dst = f"{OUTPUT_DIR}/val_images/{image_id}"
    
    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied += 1
    else:
        missing += 1
        print(f"⚠️  缺失: {image_id}")

print()
print(f"✅ 复制完成: {copied} 张图片")
if missing > 0:
    print(f"⚠️  缺失: {missing} 张图片")
print()

# 统计信息
label_counts = val_df['label'].value_counts().sort_index()
print("📊 验证集统计:")
print(f"  总样本数: {len(val_df)}")
print(f"  类别数: {val_df['label'].nunique()}")
print(f"  最小类别样本数: {label_counts.min()}")
print(f"  最大类别样本数: {label_counts.max()}")
print(f"  平均每类样本数: {label_counts.mean():.1f}")
print()

# 生成 README
with open(f"{OUTPUT_DIR}/README.txt", 'w') as f:
    f.write("验证集导出信息\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"划分参数:\n")
    f.write(f"  Val Split: {VAL_SPLIT} ({VAL_SPLIT*100:.1f}%)\n")
    f.write(f"  Random Seed: {SEED}\n")
    f.write(f"  分层采样: 是\n\n")
    f.write(f"数据统计:\n")
    f.write(f"  总样本数: {len(val_df)}\n")
    f.write(f"  类别数: {val_df['label'].nunique()}\n\n")
    f.write("文件结构:\n")
    f.write("  val.csv - 标签文件 (image_id, label)\n")
    f.write("  val_images/ - 图片目录\n")

print("✅ 生成元数据: README.txt")
print()

print("=" * 70)
print("🎉 导出完成！")
print("=" * 70)

EOF

# 打包
echo ""
echo "📦 打包验证集..."
cd exported_val_dataset
tar -czf ../val_dataset.tar.gz ./*
cd ..

echo "✅ 打包完成: val_dataset.tar.gz"
echo ""
echo "📊 文件大小:"
du -h val_dataset.tar.gz
echo ""

echo "========================================================================"
echo "🚀 使用方法"
echo "========================================================================"
echo ""
echo "1️⃣  传输到目标服务器:"
echo "   scp val_dataset.tar.gz user@server:/path/to/destination/"
echo ""
echo "2️⃣  在目标服务器解压:"
echo "   tar -xzf val_dataset.tar.gz"
echo ""
echo "3️⃣  验证数据完整性:"
echo "   python -c \"import pandas as pd; df=pd.read_csv('val.csv'); print(f'样本数: {len(df)}')\""
echo ""
echo "========================================================================"
