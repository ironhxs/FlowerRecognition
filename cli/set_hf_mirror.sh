#!/bin/bash
# 设置 HuggingFace 镜像 - 解决预训练权重下载问题
# 运行方式: source set_hf_mirror.sh 或 . set_hf_mirror.sh

echo -e "\033[32mSetting HuggingFace Mirror...\033[0m"

# 设置当前会话的环境变量
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="./cache/huggingface"
export TORCH_HOME="./cache/torch"

# 创建缓存目录
mkdir -p ./cache/huggingface
mkdir -p ./cache/torch

echo -e "\033[36m✓ HF_ENDPOINT = $HF_ENDPOINT\033[0m"
echo -e "\033[36m✓ HF_HOME = $HF_HOME\033[0m"
echo -e "\033[36m✓ TORCH_HOME = $TORCH_HOME\033[0m"

echo -e "\n\033[32mMirror configured successfully!\033[0m"
echo -e "\033[33mNow you can run: python train.py ...\033[0m"
echo -e "\n\033[90m提示：要永久生效，请将以下内容添加到 ~/.bashrc：\033[0m"
echo -e "\033[90mexport HF_ENDPOINT=\"https://hf-mirror.com\"\033[0m"
