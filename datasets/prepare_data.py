#!/usr/bin/env python3
"""
数据准备脚本 - Data Preparation Script

将原始数据转换为项目所需格式：
1. 解压 train.zip 到指定文件夹
2. 转换 train_labels.csv 格式 (filename, category_id) -> (image_id, label)
3. 重新映射 category_id 到 0-99 范围（如果需要）
"""

import os
import zipfile
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def find_train_zip():
    """
    查找 train.zip 的位置（支持 datasets 或 data 文件夹）
    
    Returns:
        train.zip 的完整路径，或 None
    """
    # 优先查找 datasets 文件夹
    datasets_zip = Path("./datasets/train.zip")
    if datasets_zip.exists():
        return datasets_zip
    
    # 其次查找 data 文件夹
    data_zip = Path("./data/train.zip")
    if data_zip.exists():
        return data_zip
    
    return None


def find_train_labels_csv():
    """
    查找 train_labels.csv 的位置（支持 datasets 或 data 文件夹）
    
    Returns:
        train_labels.csv 的完整路径，或 None
    """
    # 优先查找 datasets 文件夹
    datasets_csv = Path("./datasets/train_labels.csv")
    if datasets_csv.exists():
        return datasets_csv
    
    # 其次查找 data 文件夹
    data_csv = Path("./data/train_labels.csv")
    if data_csv.exists():
        return data_csv
    
    return None


def prepare_flower_data(data_dir: str = None):
    """
    准备花卉识别数据集
    
    Args:
        data_dir: 数据目录路径。如果为 None，自动检测位置
    """
    # 自动检测数据位置
    if data_dir is None:
        train_zip = find_train_zip()
        if train_zip is None:
            print("✗ 错误: 找不到 train.zip（应在 datasets 或 data 文件夹中）")
            return False
        data_dir = str(train_zip.parent)
    
    data_path = Path(data_dir)
    
    print("=" * 60)
    print("花卉识别数据准备 / Flower Recognition Data Preparation")
    print("=" * 60)
    print(f"数据目录: {data_path}")
    
    # 步骤 1: 解压训练图片
    train_zip = data_path / "train.zip"
    train_dir = data_path / "train"
    
    if not train_dir.exists():
        if train_zip.exists():
            print(f"\n[1/3] 解压训练图片 / Extracting training images...")
            print(f"从 {train_zip} 到 {train_dir}")
            
            with zipfile.ZipFile(train_zip, 'r') as zip_ref:
                # 获取文件列表
                file_list = zip_ref.namelist()
                
                # 带进度条解压
                for file in tqdm(file_list, desc="解压中"):
                    zip_ref.extract(file, data_path)
            
            print(f"✓ 解压完成！共 {len(os.listdir(train_dir))} 个文件")
        else:
            print(f"✗ 错误: 找不到 {train_zip}")
            return False
    else:
        print(f"\n[1/3] 训练图片目录已存在: {train_dir}")
        print(f"    包含 {len(os.listdir(train_dir))} 个文件")
    
    # 步骤 2: 转换 CSV 格式
    # 查找标签文件
    labels_file = None
    if (data_path / "train_labels.csv").exists():
        labels_file = data_path / "train_labels.csv"
    elif (data_path / "train.csv").exists():
        labels_file = data_path / "train.csv"
    
    output_csv = data_path / "train.csv"
    
    print(f"\n[2/3] 转换标签文件 / Converting label file...")
    if labels_file:
        print(f"从 {labels_file} 到 {output_csv}")
    else:
        print(f"✗ 错误: 找不到 train_labels.csv 或 train.csv")
        return False
    
    if not labels_file.exists():
        print(f"✗ 错误: 找不到 {labels_file}")
        return False
    
    # 读取原始标签
    df = pd.read_csv(labels_file)
    print(f"原始数据: {len(df)} 行")
    print(f"原始列: {list(df.columns)}")
    
    # 检查 category_id 的范围
    unique_categories = sorted(df['category_id'].unique())
    print(f"\n类别统计:")
    print(f"  - 唯一类别数: {len(unique_categories)}")
    print(f"  - 类别ID范围: {min(unique_categories)} - {max(unique_categories)}")
    
    # 创建类别映射 (category_id -> 0-based label)
    category_mapping = {cat_id: idx for idx, cat_id in enumerate(unique_categories)}
    
    # 转换为项目所需格式
    new_df = pd.DataFrame({
        'image_id': df['filename'],
        'label': df['category_id'].map(category_mapping)
    })
    
    # 保存转换后的文件
    new_df.to_csv(output_csv, index=False)
    print(f"✓ 转换完成！")
    print(f"新格式列: {list(new_df.columns)}")
    print(f"标签范围: {new_df['label'].min()} - {new_df['label'].max()}")
    
    # 步骤 3: 保存类别映射信息
    mapping_file = data_path / "category_mapping.csv"
    print(f"\n[3/3] 保存类别映射 / Saving category mapping...")
    
    mapping_df = pd.DataFrame({
        'label': list(category_mapping.values()),
        'category_id': list(category_mapping.keys()),
        'chinese_name': [df[df['category_id'] == cat_id]['chinese_name'].iloc[0] 
                         for cat_id in category_mapping.keys()],
        'english_name': [df[df['category_id'] == cat_id]['english_name'].iloc[0] 
                         for cat_id in category_mapping.keys()]
    })
    
    mapping_df.to_csv(mapping_file, index=False, encoding='utf-8-sig')
    print(f"✓ 类别映射已保存到: {mapping_file}")
    
    # 显示示例
    print("\n" + "=" * 60)
    print("数据准备完成！/ Data preparation completed!")
    print("=" * 60)
    print(f"\n数据统计 / Data Statistics:")
    print(f"  - 训练图片数: {len(new_df)}")
    print(f"  - 类别数: {len(unique_categories)}")
    print(f"  - 图片目录: {train_dir}")
    print(f"  - 标签文件: {output_csv}")
    print(f"  - 类别映射: {mapping_file}")
    
    print(f"\n前5个样本 / First 5 samples:")
    print(new_df.head())
    
    print(f"\n类别分布 / Category distribution:")
    label_counts = new_df['label'].value_counts().sort_index()
    print(f"  - 最少样本数: {label_counts.min()}")
    print(f"  - 最多样本数: {label_counts.max()}")
    print(f"  - 平均样本数: {label_counts.mean():.1f}")
    
    print("\n✓ 现在可以开始训练了！/ Ready to start training!")
    print("  运行命令: python train.py")
    
    return True


def verify_data_structure(data_dir: str = "./data"):
    """
    验证数据结构是否正确
    
    Args:
        data_dir: 数据目录路径
    """
    data_path = Path(data_dir)
    
    print("\n" + "=" * 60)
    print("验证数据结构 / Verifying Data Structure")
    print("=" * 60)
    
    checks = []
    
    # 检查训练图片目录
    train_dir = data_path / "train"
    if train_dir.exists() and train_dir.is_dir():
        num_images = len([f for f in os.listdir(train_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        checks.append((True, f"✓ 训练图片目录存在: {num_images} 张图片"))
    else:
        checks.append((False, f"✗ 训练图片目录不存在: {train_dir}"))
    
    # 检查标签文件
    train_csv = data_path / "train.csv"
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        if 'image_id' in df.columns and 'label' in df.columns:
            checks.append((True, f"✓ 标签文件格式正确: {len(df)} 条记录"))
        else:
            checks.append((False, f"✗ 标签文件格式错误: 缺少必需列"))
    else:
        checks.append((False, f"✗ 标签文件不存在: {train_csv}"))
    
    # 检查类别映射文件
    mapping_file = data_path / "category_mapping.csv"
    if mapping_file.exists():
        checks.append((True, f"✓ 类别映射文件存在"))
    else:
        checks.append((False, f"✗ 类别映射文件不存在（可选）"))
    
    # 显示检查结果
    print()
    for status, message in checks:
        print(message)
    
    all_passed = all(status for status, _ in checks[:2])  # 前两个是必需的
    
    if all_passed:
        print("\n✓ 数据结构验证通过！/ Data structure verification passed!")
        return True
    else:
        print("\n✗ 数据结构验证失败 / Data structure verification failed")
        print("请运行: python prepare_data.py")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="花卉识别数据准备工具")
    parser.add_argument('--data-dir', type=str, default=None,
                       help='数据目录路径 (默认: 自动检测 datasets 或 data 文件夹)')
    parser.add_argument('--verify-only', action='store_true',
                       help='仅验证数据结构，不进行转换')
    
    args = parser.parse_args()
    
    # 如果没有指定 data-dir，自动检测
    data_dir = args.data_dir
    if data_dir is None:
        train_zip = find_train_zip()
        if train_zip:
            data_dir = str(train_zip.parent)
            print(f"自动检测到数据在: {data_dir}\n")
        else:
            print("✗ 错误: 无法找到 train.zip")
            print("请确保 train.zip 在 datasets 或 data 文件夹中")
            exit(1)
    
    if args.verify_only:
        verify_data_structure(data_dir)
    else:
        success = prepare_flower_data(data_dir)
        if success:
            verify_data_structure(data_dir)
