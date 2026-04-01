#!/bin/bash

echo "=================================="
echo "Power Dispatch Multi-Agent System"
echo "=================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 not found"
    exit 1
fi

echo "✅ Python version: $(python3 --version)"
echo ""

# Check dependencies
echo "📦 Checking dependencies..."
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Dependencies not installed, installing..."
    pip3 install -r requirements.txt
else
    echo "✅ Dependencies installed"
fi

echo ""
echo "🚀 Starting system..."
echo ""

# Run application
python3 main.py
