#!/usr/bin/env python3
"""
快速启动脚本 - Quick Start Script
提供常用命令的快捷方式
"""

import subprocess
import sys
import os


def print_banner(text):
    print("\n" + "=" * 60)
    print(text.center(60))
    print("=" * 60 + "\n")


def main():
    print_banner("花卉识别项目快速启动 / Flower Recognition Quick Start")
    
    print("请选择要执行的操作 / Please select an operation:\n")
    print("1. 验证环境 / Verify Environment")
    print("2. 验证数据 / Verify Data")
    print("3. 开始训练 (默认配置) / Start Training (Default)")
    print("4. 开始训练 (EfficientNetV2) / Start Training (EfficientNetV2)")
    print("5. 开始训练 (ConvNeXt Tiny - 快速测试) / Start Training (ConvNeXt Tiny - Fast)")
    print("6. 查看可用模型 / List Available Models")
    print("7. 启动 TensorBoard / Start TensorBoard")
    print("8. 生成预测 / Generate Predictions")
    print("9. 退出 / Exit\n")
    
    choice = input("输入选项 / Enter choice (1-9): ").strip()
    
    if choice == "1":
        print_banner("验证环境 / Verifying Environment")
        subprocess.run([sys.executable, "verify_setup.py"])
        
    elif choice == "2":
        print_banner("验证数据 / Verifying Data")
        subprocess.run([sys.executable, "prepare_data.py", "--verify-only"])
        
    elif choice == "3":
        print_banner("开始训练 - 默认配置 (ConvNeXt Base)")
        print("这将使用以下配置:")
        print("  - 模型: ConvNeXt Base (89M 参数)")
        print("  - 增强: Strong")
        print("  - Epochs: 50")
        print("  - Batch Size: 32\n")
        confirm = input("确认开始训练? (y/n): ").strip().lower()
        if confirm == 'y':
            subprocess.run([sys.executable, "train.py"])
        else:
            print("已取消")
            
    elif choice == "4":
        print_banner("开始训练 - EfficientNetV2-L")
        print("这将使用以下配置:")
        print("  - 模型: EfficientNetV2-L (120M 参数)")
        print("  - 增强: Strong")
        print("  - Epochs: 50\n")
        confirm = input("确认开始训练? (y/n): ").strip().lower()
        if confirm == 'y':
            subprocess.run([sys.executable, "train.py", "model=efficientnet_v2_l"])
        else:
            print("已取消")
            
    elif choice == "5":
        print_banner("开始训练 - ConvNeXt Tiny (快速测试)")
        print("这将使用以下配置:")
        print("  - 模型: ConvNeXt Tiny (29M 参数)")
        print("  - 增强: Medium")
        print("  - Epochs: 30")
        print("  - 适合快速验证流程\n")
        confirm = input("确认开始训练? (y/n): ").strip().lower()
        if confirm == 'y':
            subprocess.run([
                sys.executable, "train.py", 
                "model=convnext_tiny",
                "augmentation=medium",
                "training.epochs=30"
            ])
        else:
            print("已取消")
            
    elif choice == "6":
        print_banner("可用模型列表 / Available Models")
        subprocess.run([sys.executable, "cli/flower_cli.py", "models"])
        
    elif choice == "7":
        print_banner("启动 TensorBoard")
        print("TensorBoard 将在后台运行")
        print("访问: http://localhost:6006")
        print("按 Ctrl+C 停止\n")
        try:
            subprocess.run(["tensorboard", "--logdir", "results/logs"])
        except KeyboardInterrupt:
            print("\nTensorBoard 已停止")
            
    elif choice == "8":
        print_banner("生成预测 / Generate Predictions")
        checkpoint = input("输入 checkpoint 路径 (默认: results/checkpoints/best_model.pt): ").strip()
        if not checkpoint:
            checkpoint = "results/checkpoints/best_model.pt"
        
        output = input("输入输出文件名 (默认: predictions.csv): ").strip()
        if not output:
            output = "predictions.csv"
            
        tta = input("使用 TTA? (y/n, 默认: n): ").strip().lower()
        
        cmd = [sys.executable, "inference.py", "--checkpoint", checkpoint, "--output", output]
        if tta == 'y':
            cmd.append("--tta")
            
        subprocess.run(cmd)
        
    elif choice == "9":
        print("退出...")
        sys.exit(0)
        
    else:
        print("无效选项！/ Invalid choice!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作已取消 / Operation cancelled")
        sys.exit(0)
