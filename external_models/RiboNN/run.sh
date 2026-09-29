#!/bin/bash
# RiboNN 快速运行脚本（本地环境）

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agent2

# 切换到 ribonn 目录
cd .

# 检查参数
if [ "$1" == "test" ]; then
    echo "运行测试..."
    python test_docker.py
elif [ "$1" == "predict" ]; then
    echo "运行预测..."
    python predict.py
else
    echo "用法："
    echo "  ./run.sh test      - 运行环境测试"
    echo "  ./run.sh predict   - 运行预测脚本"
    echo ""
    echo "当前环境信息："
    echo "  Python: $(python --version)"
    echo "  Conda 环境: $CONDA_DEFAULT_ENV"
    echo "  工作目录: $(pwd)"
fi
