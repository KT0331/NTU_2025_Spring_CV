#!/bin/sh

# Usage: sh init.sh environment_name

# Step 1: Get the environment name argument
ENV_NAME="$1"

# Step 2: Check if the environment name is provided
if [ "$ENV_NAME" = "" ]; then
  echo "Error: Please provide a conda environment name as an argument."
  echo "Example usage: sh init.sh myenv"
  exit 1
fi

# Step 3: Create the conda environment (using Python 3.9.0 here, you can change to 3.7, 3.8, etc.)
echo "Creating conda environment: $ENV_NAME ..."
conda create -y -n "$ENV_NAME" python=3.9.0

# Step 4: Install packages from requirements.txt file
echo "Installing packages from requirements.txt ..."
conda init
conda activate "$ENV_NAME"
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch/
conda deactivate

# Step 5: Complete
echo "Installation complete! Conda environment '$ENV_NAME' has been created and all packages installed."

# # 使用方式: sh init.sh myenv

# # 1️⃣ 環境名稱參數
# ENV_NAME="$1"

# # 2️⃣ 檢查是否提供名稱
# if [ "$ENV_NAME" = "" ]; then
#   echo "❌ 請提供 conda 環境名稱作為參數。"
#   echo "✅ 範例用法： sh init.sh myenv"
#   exit 1
# fi

# # 3️⃣ 建立 conda 環境（這裡用 Python 3.9.0，你也可以改成 3.7, 3.8 等）
# echo "🚀 建立 conda 環境: $ENV_NAME ..."
# conda create -y -n "$ENV_NAME" python=3.9.0

# # 4️⃣ 安裝 requirements.txt
# echo "📦 安裝 requirements.txt ..."
# conda run -n "$ENV_NAME" pip install -r requirements.txt -f https://download.pytorch.org/whl/torch/

# # 5️⃣ 完成
# echo "✅ 完成安裝！Conda 環境 '$ENV_NAME' 已建立並安裝所有套件。"

