#!/bin/bash
# Docker 和 nvidia-docker2 安装脚本
# 请使用 sudo 权限运行此脚本

set -e

echo "========================================="
echo "开始安装 Docker 和 nvidia-docker2"
echo "========================================="

# 1. 安装依赖包
echo ""
echo "[步骤 1/7] 安装依赖包..."
sudo apt-get update
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# 2. 添加 Docker GPG 密钥
echo ""
echo "[步骤 2/7] 添加 Docker GPG 密钥..."
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 3. 添加 Docker 软件源
echo ""
echo "[步骤 3/7] 添加 Docker 软件源..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 4. 安装 Docker Engine
echo ""
echo "[步骤 4/7] 安装 Docker Engine..."
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 5. 添加当前用户到 docker 组
echo ""
echo "[步骤 5/7] 添加用户 $USER 到 docker 组..."
sudo usermod -aG docker $USER

# 6. 安装 nvidia-docker2
echo ""
echo "[步骤 6/7] 安装 nvidia-docker2..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2

# 7. 重启 Docker 服务
echo ""
echo "[步骤 7/7] 重启 Docker 服务..."
sudo systemctl restart docker

echo ""
echo "========================================="
echo "✓ Docker 安装完成！"
echo "========================================="
echo ""
echo "请执行以下操作："
echo "  1. 退出当前终端并重新登录（或运行: newgrp docker）"
echo "  2. 验证安装："
echo "     docker --version"
echo "     docker run hello-world"
echo "     docker run --rm --gpus all nvidia/cuda:11.3.1-base nvidia-smi"
echo ""
