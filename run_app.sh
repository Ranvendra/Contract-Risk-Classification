#!/bin/bash
# run_app.sh - Entry point that automatically sets up dependencies and runs the Streamlit app.

cd "$(dirname "$0")"

VENV_DIR=".venv"

echo "================================================="
echo "   Intelligent Contract Risk Classification"
echo "================================================="

# 1. Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "⚙️  Virtual environment not found. Creating one..."
    python3 -m venv $VENV_DIR
    echo "✅ Virtual environment created."
fi

# 2. Activate virtual environment
source $VENV_DIR/bin/activate

# 3. Always ensure dependencies are up-to-date
echo "📦 Verifying dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "✅ All dependencies are installed and up to date."

# 4. Run the Streamlit application
echo "🚀 Starting the LangGraph AI Contract Agent..."
echo "================================================="
streamlit run app.py
