# RiboNN Docker 使用指南

这个 Docker 配置允许你在容器化环境中运行 RiboNN TE 预测，无需手动配置环境。

## 目录结构

```
ribonn/
├── Dockerfile              # Docker 镜像配置
├── docker-compose.yml      # Docker Compose 编排文件
├── requirements.txt        # Python 依赖
├── .dockerignore          # Docker 构建忽略文件
├── predict.py             # 预测脚本
├── RiboNN-1.0.0/          # RiboNN 模型目录
├── input/                 # 输入数据目录（需要创建）
└── output/                # 输出结果目录（需要创建）
```

## 前置要求

### 1. 安装 Docker 和 nvidia-docker

```bash
# 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 添加当前用户到 docker 组（避免每次都用 sudo）
sudo usermod -aG docker $USER
newgrp docker

# 安装 nvidia-docker2 (用于 GPU 支持)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/os-release.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 验证 GPU 是否可用
docker run --rm --gpus all nvidia/cuda:11.3.1-base nvidia-smi
```

### 2. 创建输入输出目录

```bash
cd .
mkdir -p input output
```

## 使用方法

### 方式 1：使用 Docker Compose (推荐)

#### 1. 构建镜像

```bash
cd .
docker-compose build
```

#### 2. 准备输入数据

将你的输入 TSV 文件放到 `input/` 目录下，例如：

```bash
cp your_sequences.tsv input/
```

或者修改 `predict.py` 中的输入路径。

#### 3. 运行预测

```bash
# 运行默认脚本
docker-compose up

# 运行后自动删除容器
docker-compose run --rm ribonn

# 运行自定义 Python 脚本
docker-compose run --rm ribonn python your_script.py

# 进入容器交互式运行
docker-compose run --rm ribonn bash
```

#### 4. 查看结果

结果会保存在 `output/` 目录下。

### 方式 2：直接使用 Docker 命令

#### 1. 构建镜像

```bash
cd .
docker build -t ribonn-predictor:latest .
```

#### 2. 运行容器

```bash
# 基本运行
docker run --rm --gpus all \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    ribonn-predictor:latest

# 交互式运行
docker run -it --rm --gpus all \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    ribonn-predictor:latest bash

# 运行自定义脚本
docker run --rm --gpus all \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/your_script.py:/app/your_script.py \
    ribonn-predictor:latest python your_script.py
```

## 修改预测脚本

如果你想修改 `predict.py` 并在容器中测试：

### 方法 1：重新构建镜像

```bash
# 编辑 predict.py
vim predict.py

# 重新构建
docker-compose build

# 运行
docker-compose up
```

### 方法 2：挂载脚本（开发模式）

修改 `docker-compose.yml`，取消注释：

```yaml
volumes:
  - ./input:/app/input
  - ./output:/app/output
  - ./predict.py:/app/predict.py  # 添加这一行
```

然后直接运行：

```bash
docker-compose run --rm ribonn python predict.py
```

## 性能优化

### 1. 调整 GPU 设备

如果有多个 GPU，可以指定使用哪个：

```bash
# 使用第 0 个 GPU
docker run --rm --gpus '"device=0"' ...

# 使用多个 GPU
docker run --rm --gpus '"device=0,1"' ...
```

或者修改 `docker-compose.yml`：

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # 使用 GPU 0 和 1
```

### 2. 调整批次大小

编辑 `predict.py`，修改 batch_size：

```python
predictions = predict_using_nested_cross_validation_models(
    run_df, n_folds,
    batch_size=64,  # 增加批次大小（如果显存足够）
    num_workers=8,  # 增加数据加载线程
    input_df=df
)
```

## 常见问题

### Q1: 提示 "nvidia-smi not found" 或 GPU 不可用？

检查 nvidia-docker 是否正确安装：

```bash
docker run --rm --gpus all nvidia/cuda:11.3.1-base nvidia-smi
```

如果失败，重新安装 nvidia-docker2。

### Q2: 构建镜像失败？

- 检查网络连接（pip 下载可能需要镜像源）
- 检查磁盘空间是否充足
- 查看详细错误日志：`docker-compose build --no-cache --progress=plain`

### Q3: 容器内找不到模型文件？

确保 `RiboNN-1.0.0/models/` 目录下有模型权重文件。

### Q4: 如何使用 CPU 运行（不使用 GPU）？

修改 `docker-compose.yml`，注释掉 runtime 和 GPU 相关配置：

```yaml
# runtime: nvidia  # 注释掉
environment:
  # - NVIDIA_VISIBLE_DEVICES=all  # 注释掉
  - CUDA_VISIBLE_DEVICES=-1  # 禁用 GPU
```

或使用 CPU 版本的基础镜像（修改 Dockerfile）。

### Q5: 如何查看容器日志？

```bash
# 查看运行日志
docker-compose logs

# 实时跟踪日志
docker-compose logs -f
```

## 清理和维护

```bash
# 停止所有容器
docker-compose down

# 删除镜像
docker rmi ribonn-predictor:latest

# 清理未使用的镜像和容器（释放空间）
docker system prune -a

# 查看镜像大小
docker images | grep ribonn
```

## 高级配置

### 使用镜像加速（中国大陆）

修改 `requirements.txt`，添加 pip 镜像源：

```bash
pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或在 Dockerfile 中配置：

```dockerfile
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 多阶段构建（减小镜像体积）

如果需要更小的镜像，可以使用多阶段构建（高级用户）。

## 示例工作流

```bash
# 1. 构建镜像
cd .
docker-compose build

# 2. 准备数据
echo -e "tx_id\tutr5_sequence\tcds_sequence\tutr3_sequence" > input/test.tsv
echo -e "seq1\tGGGAG\tATGAAATAG\tTGA" >> input/test.tsv

# 3. 运行预测
docker-compose run --rm ribonn python -c "
from predict import run_ribonn_prediction
import pandas as pd

df = pd.read_csv('/app/input/test.tsv', sep='\t')
result = run_ribonn_prediction(df)
result.to_csv('/app/output/result.tsv', sep='\t', index=False)
print('Done!')
"

# 4. 查看结果
cat output/result.tsv
```

## 技术支持

- 如有问题，请检查容器日志：`docker-compose logs`
- 进入容器调试：`docker-compose run --rm ribonn bash`
- 查看 RiboNN 官方文档：https://github.com/Sanofi-GitHub/RiboNN
