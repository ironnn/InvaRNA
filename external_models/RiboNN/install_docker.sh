#!/bin/bash
# Docker 安装脚本

echo "开始安装 Docker..."

# 1. 更新包索引
sudo apt-get update

# 2. 安装依赖
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# 3. 添加 Docker 官方 GPG 密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 4. 设置稳定版仓库
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5. 安装 Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 6. 将当前用户添加到 docker 组
sudo usermod -aG docker $USER

# 7. 安装 nvidia-docker2（GPU 支持）
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# 8. 重启 Docker 服务
sudo systemctl restart docker

echo "Docker 安装完成！"
echo "请运行以下命令使组权限生效："
echo "  newgrp docker"
echo ""
echo "然后测试安装："
echo "  docker --version"
echo "  docker run hello-world"
echo "  docker run --rm --gpus all nvidia/cuda:11.3.1-base nvidia-smi"
