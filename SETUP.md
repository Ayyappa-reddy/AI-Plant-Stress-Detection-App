# 🚀 Quick Setup Guide

## Prerequisites
- Python 3.8+ 
- Node.js 16+
- Your plant dataset in `Data_Set/PlantVillage/`

## 🎯 One-Command Setup

### Option 1: Automated Setup (Recommended)
```bash
# Make scripts executable and run training
chmod +x train.sh start.sh
./train.sh
```

### Option 2: Manual Setup
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Train the AI model
cd training
python train_model.py

# 3. Start the backend
cd ../backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 4. Start the frontend (in new terminal)
cd ../frontend
npm install
npm run dev
```

## 🌐 Access Your App
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📁 Project Structure
```
├── backend/                 # FastAPI backend
├── frontend/                # Next.js frontend
├── training/                # Model training scripts
├── Data_Set/               # Your plant dataset
├── train.sh                 # Training script
├── start.sh                 # Startup script
└── README.md               # Detailed documentation
```

## 🔧 Troubleshooting

### Model Training Issues
- Ensure your dataset is in the correct location
- Check Python dependencies are installed
- Verify you have enough disk space

### Backend Issues
- Check if port 8000 is available
- Ensure model file exists in `backend/models/`
- Check Python environment

### Frontend Issues
- Check if port 3000 is available
- Ensure Node.js dependencies are installed
- Check browser console for errors

## 📞 Need Help?
Check the main README.md for detailed documentation and troubleshooting steps.
