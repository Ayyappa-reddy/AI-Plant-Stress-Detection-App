#!/bin/bash

echo "🌱 Starting AI Plant Disease Model Training..."
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "plant_ai_env" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv plant_ai_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source plant_ai_env/bin/activate

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not available in virtual environment. Please check the setup."
    exit 1
fi

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not available in virtual environment. Please check the setup."
    exit 1
fi

# Install requirements
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies. Please check your Python environment."
    exit 1
fi

# Check if dataset exists
if [ ! -d "Data_Set/PlantVillage" ]; then
    echo "❌ Dataset not found at Data_Set/PlantVillage/"
    echo "Please ensure your plant dataset is in the correct location."
    exit 1
fi

# Create models directory
echo "📁 Creating models directory..."
mkdir -p backend/models

# Start training
echo "🚀 Starting model training..."
echo "This may take 1-2 hours depending on your hardware..."
echo ""

cd training
python train_model.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo "📁 Model saved to: backend/models/plant_model.pth"
    echo "📊 Training history saved to: backend/models/training_history.png"
    echo ""
    echo "🎉 You can now start the backend server!"
    echo "Run: cd backend && source ../plant_ai_env/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000"
else
    echo ""
    echo "❌ Training failed. Please check the error messages above."
    exit 1
fi
