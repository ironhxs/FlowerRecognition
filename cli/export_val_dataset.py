"""
导出验证集数据

这个脚本会：
1. 使用和训练时相同的 seed 和 val_split 参数
2. 生成 val.csv (包含 image_id 和 label)
3. 可选：复制验证集图片到独立目录
"""

import os
import sys
from pathlib import Path
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))


@hydra.main(version_base=None, config_path="configs", config_name="swin_v2_anti_overfit")
def export_val_dataset(cfg: DictConfig):
    """导出验证集数据"""
    
    print("=" * 70)
    print("📦 导出验证集数据")
    print("=" * 70)
    print()
    
    # 读取训练标签文件
    train_csv = cfg.dataset.train_csv
    if not os.path.exists(train_csv):
        print(f"❌ 错误: 找不到训练标签文件 {train_csv}")
        return
    
    df = pd.read_csv(train_csv)
    print(f"✅ 加载训练数据: {len(df)} 个样本")
    print(f"   列名: {list(df.columns)}")
    print()
    
    # 使用和训练时相同的参数进行划分
    val_split = cfg.dataset.val_split
    seed = cfg.seed
    
    print(f"📊 划分参数:")
    print(f"   Val Split: {val_split} ({val_split*100:.1f}%)")
    print(f"   Random Seed: {seed}")
    print()
    
    # 分层划分
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        df['image_id'].tolist(),
        df['label'].tolist(),
        test_size=val_split,
        random_state=seed,
        stratify=df['label'].tolist()
    )
    
    print(f"✅ 划分完成:")
    print(f"   Train Set: {len(train_ids)} 个样本")
    print(f"   Val Set:   {len(val_ids)} 个样本")
    print()
    
    # 创建验证集 DataFrame
    val_df = pd.DataFrame({
        'image_id': val_ids,
        'label': val_labels
    })
    
    # 按 image_id 排序（便于查看）
    val_df = val_df.sort_values('image_id').reset_index(drop=True)
    
    # 输出目录
    output_dir = Path("./exported_val_dataset")
    output_dir.mkdir(exist_ok=True)
    
    # 保存 val.csv
    val_csv_path = output_dir / "val.csv"
    val_df.to_csv(val_csv_path, index=False)
    print(f"✅ 保存验证集标签: {val_csv_path}")
    print(f"   样本数: {len(val_df)}")
    print(f"   类别分布统计:")
    label_counts = val_df['label'].value_counts().sort_index()
    print(f"   - 最小类别样本数: {label_counts.min()}")
    print(f"   - 最大类别样本数: {label_counts.max()}")
    print(f"   - 平均每类样本数: {label_counts.mean():.1f}")
    print()
    
    # 询问是否复制图片
    print("❓ 是否复制验证集图片到独立目录？")
    print("   (这会占用额外磁盘空间，但方便迁移到其他服务器)")
    copy_images = input("   输入 'y' 复制图片，其他键跳过: ").strip().lower()
    
    if copy_images == 'y':
        # 创建验证集图片目录
        val_images_dir = output_dir / "val_images"
        val_images_dir.mkdir(exist_ok=True)
        
        train_dir = Path(cfg.dataset.train_dir)
        
        print()
        print(f"📁 复制验证集图片...")
        copied = 0
        missing = 0
        
        for image_id in tqdm(val_ids, desc="复制中"):
            src_path = train_dir / image_id
            
            if src_path.exists():
                dst_path = val_images_dir / image_id
                shutil.copy2(src_path, dst_path)
                copied += 1
            else:
                missing += 1
                print(f"   ⚠️  找不到图片: {image_id}")
        
        print()
        print(f"✅ 图片复制完成:")
        print(f"   成功: {copied} 张")
        if missing > 0:
            print(f"   ⚠️  缺失: {missing} 张")
        print(f"   目录: {val_images_dir}")
        print()
    
    # 生成元数据文件
    metadata_path = output_dir / "README.txt"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write("验证集导出信息\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"导出时间: {pd.Timestamp.now()}\n")
        f.write(f"配置文件: swin_v2_anti_overfit.yaml\n\n")
        f.write(f"划分参数:\n")
        f.write(f"  - Val Split: {val_split} ({val_split*100:.1f}%)\n")
        f.write(f"  - Random Seed: {seed}\n")
        f.write(f"  - 分层采样: 是 (stratify by label)\n\n")
        f.write(f"数据统计:\n")
        f.write(f"  - 总样本数: {len(val_df)}\n")
        f.write(f"  - 类别数: {val_df['label'].nunique()}\n")
        f.write(f"  - 最小类别样本数: {label_counts.min()}\n")
        f.write(f"  - 最大类别样本数: {label_counts.max()}\n")
        f.write(f"  - 平均每类样本数: {label_counts.mean():.1f}\n\n")
        f.write(f"文件列表:\n")
        f.write(f"  - val.csv: 验证集标签文件 (image_id, label)\n")
        if copy_images == 'y':
            f.write(f"  - val_images/: 验证集图片目录\n")
        f.write("\n")
        f.write("使用方法:\n")
        f.write("  1. 将整个 exported_val_dataset 目录复制到目标服务器\n")
        f.write("  2. 在目标服务器上运行推理验证准确率\n")
        f.write("  3. 对比不同环境的结果差异\n")
    
    print(f"✅ 保存元数据: {metadata_path}")
    print()
    
    # 打包建议
    print("=" * 70)
    print("🎉 导出完成！")
    print("=" * 70)
    print()
    print("📦 打包命令 (用于传输到其他服务器):")
    print(f"   cd {output_dir.parent}")
    print(f"   tar -czf val_dataset.tar.gz {output_dir.name}/")
    print()
    print("📤 传输到目标服务器:")
    print(f"   scp val_dataset.tar.gz user@server:/path/to/destination/")
    print()
    print("📂 解压:")
    print(f"   tar -xzf val_dataset.tar.gz")
    print()
    print("=" * 70)


if __name__ == "__main__":
    export_val_dataset()
