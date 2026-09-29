#!/bin/bash
# Docker 安装脚本 - 使用阿里云镜像源
# 适用于国内网络环境

set -e

echo "========================================="
echo "使用阿里云镜像安装 Docker"
echo "========================================="

# 1. 卸载旧版本（如果存在）
echo ""
echo "[步骤 1/6] 卸载旧版本 Docker（如果有）..."
sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

# 2. 安装依赖
echo ""
echo "[步骤 2/6] 安装依赖包..."
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# 3. 添加 Docker 的官方 GPG 密钥（使用阿里云镜像）
echo ""
echo "[步骤 3/6] 添加 Docker GPG 密钥（阿里云镜像）..."
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 4. 设置 Docker 仓库（阿里云镜像）
echo ""
echo "[步骤 4/6] 添加 Docker 软件源（阿里云镜像）..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. 安装 Docker Engine
echo ""
echo "[步骤 5/6] 安装 Docker Engine..."
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 6. 配置 Docker（镜像加速 + 用户权限）
echo ""
echo "[步骤 6/6] 配置 Docker..."

# 添加用户到 docker 组
sudo usermod -aG docker $USER

# 配置镜像加速器（阿里云）
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": [
    "https://mirror.ccs.tencentyun.com",
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com"
  ]
}
EOF

# 重启 Docker
sudo systemctl daemon-reload
sudo systemctl restart docker
sudo systemctl enable docker

echo ""
echo "========================================="
echo "✓ Docker 安装完成！"
echo "========================================="
echo ""
echo "Docker 版本："
docker --version
echo ""
echo "请执行以下操作："
echo "  1. 退出当前终端并重新登录（或运行: newgrp docker）"
echo "  2. 测试 Docker："
echo "     docker run hello-world"
echo ""
echo "如果需要 GPU 支持，请继续运行："
echo "  ./install_nvidia_docker.sh"
echo ""
