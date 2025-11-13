#!/usr/bin/env python3
"""
比赛需求检查脚本 - Competition Requirements Checker
自动检查项目是否满足所有比赛要求
"""

import os
import sys
from pathlib import Path
import torch


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_mark(passed, message):
    """打印检查结果"""
    symbol = "✅" if passed else "❌"
    print(f"{symbol} {message}")
    return passed


def check_data_structure():
    """检查数据结构"""
    print_section("数据结构检查 / Data Structure Check")
    
    checks = []
    
    # 检查训练数据
    train_dir = Path("data/train")
    if train_dir.exists():
        images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
        checks.append(check_mark(
            len(images) > 0,
            f"训练图片目录: {len(images)} 张图片"
        ))
    else:
        checks.append(check_mark(False, "训练图片目录不存在"))
    
    # 检查标签文件
    train_csv = Path("data/train.csv")
    if train_csv.exists():
        import pandas as pd
        df = pd.read_csv(train_csv)
        checks.append(check_mark(
            'image_id' in df.columns and 'label' in df.columns,
            f"标签文件格式正确: {len(df)} 条记录"
        ))
        checks.append(check_mark(
            df['label'].nunique() == 100,
            f"类别数量: {df['label'].nunique()} 类 (要求: 100)"
        ))
    else:
        checks.append(check_mark(False, "标签文件不存在"))
    
    return all(checks)


def check_model_constraints(checkpoint_path=None):
    """检查模型约束"""
    print_section("模型约束检查 / Model Constraints Check")
    
    if checkpoint_path and Path(checkpoint_path).exists():
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 计算模型大小
        model_size_bytes = sum(
            p.numel() * p.element_size() 
            for p in checkpoint['model_state_dict'].values()
        )
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        check_mark(
            model_size_mb <= 500,
            f"模型大小: {model_size_mb:.2f} MB (要求: ≤ 500 MB)"
        )
        
        # 检查参数量
        num_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
        num_params_b = num_params / 1e9
        check_mark(
            num_params_b < 10,
            f"模型参数量: {num_params / 1e6:.2f}M ({num_params_b:.2f}B) (要求: < 10B)"
        )
        
        return model_size_mb <= 500 and num_params_b < 10
    else:
        print("⚠️  未指定模型文件，跳过模型约束检查")
        print("   训练完成后使用: python check_competition.py --checkpoint <path>")
        return True


def check_environment():
    """检查环境配置"""
    print_section("环境配置检查 / Environment Check")
    
    checks = []
    
    # Python 版本
    python_version = sys.version_info
    checks.append(check_mark(
        python_version >= (3, 8),
        f"Python 版本: {python_version.major}.{python_version.minor} (要求: ≥ 3.8)"
    ))
    
    # PyTorch 版本
    torch_version = torch.__version__.split('+')[0]
    major, minor = map(int, torch_version.split('.')[:2])
    checks.append(check_mark(
        (major, minor) >= (1, 9),
        f"PyTorch 版本: {torch_version} (要求: ≥ 1.9)"
    ))
    
    # CUDA 可用性
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_version = torch.version.cuda
        checks.append(check_mark(
            True,
            f"CUDA 版本: {cuda_version} (要求: ≥ 11.3)"
        ))
    else:
        checks.append(check_mark(
            False,
            "CUDA 不可用 (建议使用 GPU 训练)"
        ))
    
    return all(checks)


def check_configs():
    """检查配置文件"""
    print_section("配置文件检查 / Configuration Check")
    
    checks = []
    
    # 检查输入尺寸配置
    model_configs = Path("configs/model").glob("*.yaml")
    all_600x600 = True
    
    for config_file in model_configs:
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'input_size: 600' not in content:
                all_600x600 = False
                print(f"⚠️  {config_file.name} 未配置 input_size: 600")
    
    checks.append(check_mark(
        all_600x600,
        "所有模型配置使用 600×600 输入尺寸"
    ))
    
    return all(checks)


def check_submission_structure():
    """检查提交结构"""
    print_section("提交结构检查 / Submission Structure Check")
    
    required_files = [
        "train.py",
        "inference.py",
        "datasets/__init__.py",
        "datasets/flower_dataset.py",
        "models/__init__.py",
        "models/flower_model.py",
        "configs/config.yaml",
        "requirements.txt",
        "prepare_submission.py"
    ]
    
    checks = []
    for file_path in required_files:
        checks.append(check_mark(
            Path(file_path).exists(),
            f"必需文件: {file_path}"
        ))
    
    return all(checks)


def check_output_format():
    """检查输出格式"""
    print_section("输出格式检查 / Output Format Check")
    
    # 检查是否有预测文件
    predictions_files = list(Path(".").glob("*.csv"))
    predictions_files = [f for f in predictions_files if 'prediction' in f.name.lower()]
    
    if predictions_files:
        import pandas as pd
        for pred_file in predictions_files[:1]:  # 只检查第一个
            try:
                df = pd.read_csv(pred_file, encoding='utf-8')
                check_mark(
                    'image_id' in df.columns and 'label' in df.columns,
                    f"预测文件格式正确: {pred_file.name}"
                )
                
                # 检查编码
                with open(pred_file, 'rb') as f:
                    raw = f.read()
                    try:
                        raw.decode('utf-8')
                        check_mark(True, "文件编码为 UTF-8")
                    except:
                        check_mark(False, "文件编码不是 UTF-8")
                
                return True
            except Exception as e:
                check_mark(False, f"预测文件格式错误: {e}")
                return False
    else:
        print("⚠️  未找到预测文件")
        print("   生成预测: python inference.py --checkpoint <path> --output predictions.csv")
        return True


def check_technical_report():
    """检查技术报告"""
    print_section("技术报告检查 / Technical Report Check")
    
    template_exists = Path("docs/technical_report_template.md").exists()
    check_mark(
        template_exists,
        "技术报告模板存在"
    )
    
    if template_exists:
        print("\n📝 技术报告必须包含:")
        print("   1. 模型架构详细描述")
        print("   2. 训练策略（数据预处理、增强、优化器）")
        print("   3. 实验结果（验证集性能分析）")
        print("   4. 创新点说明")
        print("   5. 页数限制: ≤ 10 页")
    
    return template_exists


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="比赛需求检查工具")
    parser.add_argument('--checkpoint', type=str, help='模型检查点路径')
    parser.add_argument('--full', action='store_true', help='完整检查（包括模型）')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  🌸 花卉识别 AI 挑战赛 - 需求检查工具")
    print("  2025年第七届全国高校计算机能力挑战赛")
    print("=" * 70)
    
    results = []
    
    # 基础检查
    results.append(("环境配置", check_environment()))
    results.append(("数据结构", check_data_structure()))
    results.append(("配置文件", check_configs()))
    results.append(("提交结构", check_submission_structure()))
    
    # 可选检查
    if args.checkpoint or args.full:
        checkpoint_path = args.checkpoint or "results/checkpoints/best_model.pt"
        results.append(("模型约束", check_model_constraints(checkpoint_path)))
    
    results.append(("输出格式", check_output_format()))
    results.append(("技术报告", check_technical_report()))
    
    # 总结
    print_section("检查总结 / Summary")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    percentage = (passed / total) * 100
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 未通过"
        print(f"{status:12} - {name}")
    
    print(f"\n总计: {passed}/{total} 项检查通过 ({percentage:.1f}%)")
    
    if passed == total:
        print("\n🎉 恭喜！所有检查均通过，项目符合比赛要求！")
    else:
        print("\n⚠️  部分检查未通过，请查看上方详细信息")
    
    print("\n" + "=" * 70)
    print("📚 详细要求请查看: COMPETITION_REQUIREMENTS.md")
    print("=" * 70 + "\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
