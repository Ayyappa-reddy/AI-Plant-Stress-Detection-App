#!/bin/bash

echo "🌱 Starting AI Plant Stress Detection Web App..."
echo "================================================"

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        return 0
    else
        return 1
    fi
}

# Check if backend port is available
if check_port 8000; then
    echo "❌ Port 8000 is already in use. Please stop the service using port 8000 first."
    exit 1
fi

# Check if frontend port is available
if check_port 3000; then
    echo "❌ Port 3000 is already in use. Please stop the service using port 3000 first."
    exit 1
fi

# Check if model exists
if [ ! -f "backend/models/plant_model.pth" ]; then
    echo "❌ AI model not found. Please train the model first by running:"
    echo "   chmod +x train.sh && ./train.sh"
    exit 1
fi

# Start backend
echo "🚀 Starting FastAPI backend..."
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Check if backend is running
if ! check_port 8000; then
    echo "❌ Backend failed to start. Please check the logs."
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "✅ Backend started successfully on http://localhost:8000"

# Start frontend
echo "🚀 Starting Next.js frontend..."
cd ../frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
echo "⏳ Waiting for frontend to start..."
sleep 10

# Check if frontend is running
if ! check_port 3000; then
    echo "❌ Frontend failed to start. Please check the logs."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 1
fi

echo "✅ Frontend started successfully on http://localhost:3000"
echo ""
echo "🎉 AI Plant Stress Detection Web App is now running!"
echo ""
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services..."

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "✅ Services stopped. Goodbye!"
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

# Wait for user to stop
wait
