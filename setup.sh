#!/bin/bash

echo "======================================"
echo "動画フォーム比較分析システム セットアップ"
echo "======================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}[1/5] Pythonバージョン確認...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python $python_version が検出されました"
echo ""

# Create virtual environment
echo -e "${YELLOW}[2/5] 仮想環境の作成...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ 仮想環境を作成しました${NC}"
else
    echo -e "${GREEN}✓ 仮想環境は既に存在します${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${YELLOW}[3/5] 仮想環境のアクティベート...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ 仮想環境をアクティベートしました${NC}"
echo ""

# Install dependencies
echo -e "${YELLOW}[4/5] 依存関係のインストール...${NC}"
pip install --upgrade pip
pip install -r requirements.txt --break-system-packages
echo -e "${GREEN}✓ 依存関係をインストールしました${NC}"
echo ""

# Check Ollama
echo -e "${YELLOW}[5/5] Ollamaの確認...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollamaがインストールされています${NC}"
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollamaサーバーが起動しています${NC}"
    else
        echo -e "${YELLOW}⚠ Ollamaサーバーが起動していません${NC}"
        echo "別のターミナルで 'ollama serve' を実行してください"
    fi
    
    # Check if llama3.2 is available
    if ollama list | grep -q "llama3.2"; then
        echo -e "${GREEN}✓ llama3.2 モデルが利用可能です${NC}"
    else
        echo -e "${YELLOW}⚠ llama3.2 モデルが見つかりません${NC}"
        echo "以下のコマンドでダウンロードしてください:"
        echo "  ollama pull llama3.2"
    fi
else
    echo -e "${RED}✗ Ollamaがインストールされていません${NC}"
    echo "以下のURLからインストールしてください:"
    echo "  https://ollama.ai"
    echo ""
    echo "インストール後、以下のコマンドを実行:"
    echo "  ollama pull llama3.2"
    echo "  ollama serve"
fi
echo ""

# Create necessary directories
echo "必要なディレクトリを作成..."
mkdir -p uploads outputs static models
echo -e "${GREEN}✓ ディレクトリを作成しました${NC}"
echo ""

# Setup complete
echo "======================================"
echo -e "${GREEN}セットアップ完了！${NC}"
echo "======================================"
echo ""
echo "次のステップ:"
echo "1. Ollamaが起動していない場合:"
echo "   ollama serve"
echo ""
echo "2. アプリケーションの起動:"
echo "   python app.py"
echo ""
echo "3. ブラウザで以下にアクセス:"
echo "   http://localhost:5001"
echo ""
echo -e "${YELLOW}注意: VitPoseの完全な機能を使用するには、${NC}"
echo -e "${YELLOW}追加のモデルファイルが必要です。${NC}"
echo -e "${YELLOW}README.mdを参照してください。${NC}"
echo ""
