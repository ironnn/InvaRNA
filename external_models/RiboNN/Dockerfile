# 使用 NVIDIA CUDA 基础镜像 (与 RiboNN 要求的 CUDA 11.3 匹配)
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# 设置环境变量，避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 创建 python -> python3.10 的软链接
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 升级 pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY RiboNN-1.0.0/ ./RiboNN-1.0.0/
COPY predict.py .

# 创建输入输出目录
RUN mkdir -p /app/input /app/output

# 设置默认命令
CMD ["python", "predict.py"]
