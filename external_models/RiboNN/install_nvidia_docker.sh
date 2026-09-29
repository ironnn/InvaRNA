#!/bin/bash
# nvidia-docker2 安装脚本
# 在安装完 Docker 后运行此脚本

set -e

echo "========================================="
echo "安装 nvidia-docker2（GPU 支持）"
echo "========================================="

# 检查 Docker 是否已安装
if ! command -v docker &> /dev/null; then
    echo "错误: 请先安装 Docker！"
    echo "运行: ./install_docker_cn.sh"
    exit 1
fi

echo ""
echo "[步骤 1/3] 添加 NVIDIA Container Toolkit 仓库..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo ""
echo "[步骤 2/3] 安装 nvidia-docker2..."
sudo apt-get update
sudo apt-get install -y nvidia-docker2

echo ""
echo "[步骤 3/3] 重启 Docker 服务..."
sudo systemctl restart docker

echo ""
echo "========================================="
echo "✓ nvidia-docker2 安装完成！"
echo "========================================="
echo ""
echo "测试 GPU 是否可用："
echo "  docker run --rm --gpus all nvidia/cuda:11.3.1-base nvidia-smi"
echo ""
