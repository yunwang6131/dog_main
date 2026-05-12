#!/bin/bash

set -e

# 获取当前脚本所在目录
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 自动推断 IsaacLab 路径：脚本目录 -> 上级目录 -> 再拼接 /IsaacLab
ISAACLAB_PATH="$(realpath "$CURRENT_DIR/../IsaacLab")"

# 如果 IsaacLab 目录不存在则报错
if [ ! -d "$ISAACLAB_PATH" ]; then
    echo "Error: IsaacLab directory not found at expected location: $ISAACLAB_PATH"
    echo "Make sure IsaacLab and isaaclab_plugin are in the same parent folder."
    exit 1
fi

echo "Automatically detected IsaacLab path: $ISAACLAB_PATH"

echo "Copying action manager..."
cp "$CURRENT_DIR/action_manager.py" "$ISAACLAB_PATH/source/isaaclab/isaaclab/managers/"

echo "Copying reward manager..."
cp "$CURRENT_DIR/reward_manager.py" "$ISAACLAB_PATH/source/isaaclab/isaaclab/managers/"

echo "Copying custom exporter..."
cp "$CURRENT_DIR/exporter.py" "$ISAACLAB_PATH/source/isaaclab_rl/isaaclab_rl/rsl_rl/"