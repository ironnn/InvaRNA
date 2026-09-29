#!/bin/bash
# RiboNN Docker 完整构建和测试脚本
# 请确保已运行: newgrp docker

set -e

cd .

echo "========================================="
echo "RiboNN Docker 构建和测试"
echo "========================================="

# 1. 检查 Docker 权限
echo ""
echo "[步骤 1/6] 检查 Docker 权限..."
if ! docker ps &>/dev/null; then
    echo "❌ Docker 权限不足，请运行: newgrp docker"
    exit 1
fi
echo "✓ Docker 权限正常"

# 2. 测试 Docker
echo ""
echo "[步骤 2/6] 测试 Docker..."
docker run --rm hello-world
echo "✓ Docker 运行正常"

# 3. 检查并安装 nvidia-docker2（如果需要）
echo ""
echo "[步骤 3/6] 检查 GPU 支持..."
if docker run --rm --gpus all nvidia/cuda:11.3.1-base nvidia-smi &>/dev/null; then
    echo "✓ GPU 支持已启用"
else
    echo "⚠ GPU 支持未启用，尝试安装 nvidia-docker2..."
    if [ -f "./install_nvidia_docker.sh" ]; then
        ./install_nvidia_docker.sh
    else
        echo "请手动安装: sudo ./install_nvidia_docker.sh"
        exit 1
    fi
fi

# 4. 构建 Docker 镜像
echo ""
echo "[步骤 4/6] 构建 RiboNN Docker 镜像..."
echo "这可能需要几分钟时间..."
docker-compose build

# 5. 测试镜像
echo ""
echo "[步骤 5/6] 测试 Docker 镜像..."
docker-compose run --rm ribonn python test_docker.py

# 6. 显示镜像信息
echo ""
echo "[步骤 6/6] 镜像信息..."
docker images | grep ribonn

echo ""
echo "========================================="
echo "✓ 所有步骤完成！"
echo "========================================="
echo ""
echo "使用方法："
echo "  1. 测试环境:"
echo "     docker-compose run --rm ribonn python test_docker.py"
echo ""
echo "  2. 运行预测:"
echo "     docker-compose run --rm ribonn python predict.py"
echo ""
echo "  3. 交互式运行:"
echo "     docker-compose run --rm ribonn bash"
echo ""
