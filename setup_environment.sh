#!/bin/bash

# Setup script for Iris Ensemble Learning Project
# Author: Francesco
# Date: October 2025

echo "🚀 Setting up Iris Ensemble Learning Environment"
echo "================================================"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "📚 Installing required packages..."
pip install scikit-learn pandas numpy matplotlib seaborn

# Verify installation
echo "✅ Verifying installation..."
python -c "
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print(f'✅ scikit-learn: {sklearn.__version__}')
print(f'✅ pandas: {pd.__version__}')
print(f'✅ numpy: {np.__version__}')
print(f'✅ matplotlib: {matplotlib.__version__}')
print(f'✅ seaborn: {sns.__version__}')
"

echo ""
echo "🎉 Environment setup completed successfully!"
echo "To activate the environment, run: source .venv/bin/activate"
echo "To run the analysis, execute: python src/main_iris_ensemble.py"