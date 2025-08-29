from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Dict, Any
from pydantic import BaseModel
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Define the model class here to avoid import issues
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PlantDiseaseModel, self).__init__()
        
        # Load pretrained MobileNetV2 - use weights parameter for newer PyTorch versions
        try:
            # Try new API first (PyTorch 1.13+)
            self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        except AttributeError:
            # Fallback to old API
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Freeze early layers for transfer learning
        for param in self.backbone.features[:10].parameters():
            param.requires_grad = False
        
        # Modify the classifier for our number of classes
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

app = FastAPI(
    title="AI Plant Stress Detection API",
    description="API for detecting plant diseases and stress conditions from leaf images",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
class_names = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Contact form model
class ContactForm(BaseModel):
    name: str
    email: str
    message: str

# Email configuration
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "ayyappareddyyennam@gmail.com",
    "sender_password": "klqo xvqq frxm nhiv",  # You'll need to generate this
    "recipient_email": "ayyappareddyyennam@gmail.com"
}

# Disease recommendations
DISEASE_RECOMMENDATIONS = {
    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial spot is a serious disease that can cause significant yield loss.",
        "treatment": "Remove infected plants, apply copper-based bactericides, and practice crop rotation.",
        "prevention": "Use disease-free seeds, avoid overhead irrigation, and maintain proper spacing."
    },
    "Pepper__bell___healthy": {
        "description": "Your pepper plant appears to be healthy with no visible disease symptoms.",
        "treatment": "No treatment needed. Continue with regular care and monitoring.",
        "prevention": "Maintain good growing conditions, proper watering, and regular inspection."
    },
    "Potato___Early_blight": {
        "description": "Early blight is a common fungal disease that affects potato leaves and stems.",
        "treatment": "Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves.",
        "prevention": "Ensure proper spacing, avoid overhead irrigation, and rotate crops."
    },
    "Potato___Late_blight": {
        "description": "Late blight is a devastating disease that can destroy entire potato crops quickly.",
        "treatment": "Apply fungicides immediately. Remove and destroy infected plants.",
        "prevention": "Plant resistant varieties, avoid overhead irrigation, and monitor weather conditions."
    },
    "Potato___healthy": {
        "description": "Your potato plant appears to be healthy with no visible disease symptoms.",
        "treatment": "No treatment needed. Continue with regular care and monitoring.",
        "prevention": "Maintain good growing conditions, proper watering, and regular inspection."
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial spot causes dark, water-soaked lesions on tomato leaves and fruits.",
        "treatment": "Remove infected plants, apply copper-based bactericides, and practice crop rotation.",
        "prevention": "Use disease-free seeds, avoid overhead irrigation, and maintain proper spacing."
    },
    "Tomato_Early_blight": {
        "description": "Early blight causes dark brown spots with concentric rings on tomato leaves.",
        "treatment": "Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves.",
        "prevention": "Ensure proper spacing, avoid overhead irrigation, and rotate crops."
    },
    "Tomato_Late_blight": {
        "description": "Late blight is a serious disease that can destroy tomato plants rapidly.",
        "treatment": "Apply fungicides immediately. Remove and destroy infected plants.",
        "prevention": "Plant resistant varieties, avoid overhead irrigation, and monitor weather conditions."
    },
    "Tomato_Leaf_Mold": {
        "description": "Leaf mold is a fungal disease that thrives in humid conditions.",
        "treatment": "Improve air circulation, reduce humidity, and apply fungicides if necessary.",
        "prevention": "Ensure proper spacing, avoid overhead irrigation, and maintain good ventilation."
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Septoria leaf spot causes small, dark spots with gray centers on tomato leaves.",
        "treatment": "Remove infected leaves and apply fungicides containing chlorothalonil.",
        "prevention": "Avoid overhead irrigation, maintain proper spacing, and rotate crops."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Spider mites are tiny pests that can cause significant damage to tomato plants.",
        "treatment": "Apply insecticidal soap or neem oil. Use predatory mites for biological control.",
        "prevention": "Monitor regularly, maintain proper humidity, and avoid over-fertilization."
    },
    "Tomato__Target_Spot": {
        "description": "Target spot causes dark brown spots with target-like rings on tomato leaves.",
        "treatment": "Remove infected leaves and apply fungicides containing chlorothalonil.",
        "prevention": "Ensure proper spacing, avoid overhead irrigation, and rotate crops."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "This viral disease causes yellowing and curling of tomato leaves.",
        "treatment": "Remove infected plants immediately. Control whitefly populations.",
        "prevention": "Use virus-resistant varieties, control whiteflies, and remove weeds."
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Tomato mosaic virus causes mottled leaves and stunted growth.",
        "treatment": "Remove infected plants immediately. Disinfect tools and hands.",
        "prevention": "Use virus-free seeds, control aphids, and practice good hygiene."
    },
    "Tomato_healthy": {
        "description": "Your tomato plant appears to be healthy with no visible disease symptoms.",
        "treatment": "No treatment needed. Continue with regular care and monitoring.",
        "prevention": "Maintain good growing conditions, proper watering, and regular inspection."
    }
}

def load_model():
    """Load the trained model and class names"""
    global model, class_names
    
    model_path = "models/plant_model.pth"
    class_names_path = "models/class_names.json"
    
    try:
        # Load class names
        with open(class_names_path, 'r') as f:
            import json
            class_names = json.load(f)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        num_classes = checkpoint['num_classes']
        
        model = PlantDiseaseModel(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
        
        print(f"Model loaded successfully with {num_classes} classes")
        print(f"Class names: {class_names}")
        
    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        raise e
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess image for model inference"""
    try:
        # Load image
        import io
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply same transforms as validation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transform image
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.to(device)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")

def predict_image(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Make prediction on preprocessed image"""
    global model, class_names
    
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get top prediction
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            
            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
            top3_predictions = []
            
            for i in range(3):
                idx = top3_indices[0][i].item()
                prob = top3_probs[0][i].item()
                top3_predictions.append({
                    "class": class_names[idx],
                    "confidence": round(prob * 100, 2)
                })
            
            # Get recommendations
            recommendations = DISEASE_RECOMMENDATIONS.get(predicted_class, {
                "description": "Unknown condition",
                "treatment": "Consult with a plant expert",
                "prevention": "Monitor plant health regularly"
            })
            
            return {
                "predicted_class": predicted_class,
                "confidence": round(confidence_score * 100, 2),
                "top3_predictions": top3_predictions,
                "recommendations": recommendations
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
        print("API started successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Plant Stress Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "num_classes": len(class_names) if class_names else 0
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict plant disease from uploaded image"""
    
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if file.size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File size too large (max 10MB)")
    
    try:
        # Read image
        image_bytes = await file.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        
        # Make prediction
        result = predict_image(image_tensor)
        
        return result
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/classes")
async def get_classes():
    """Get list of supported plant conditions"""
    return {
        "classes": class_names,
        "count": len(class_names)
    }

@app.get("/recommendations/{class_name}")
async def get_recommendations(class_name: str):
    """Get recommendations for a specific plant condition"""
    if class_name not in DISEASE_RECOMMENDATIONS:
        raise HTTPException(status_code=404, detail="Plant condition not found")
    
    return DISEASE_RECOMMENDATIONS[class_name]

def send_email(name: str, email: str, message: str) -> bool:
    """Send email notification"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG["sender_email"]
        msg['To'] = EMAIL_CONFIG["recipient_email"]
        msg['Subject'] = f"New Contact Form Submission from {name}"
        
        # Email body
        body = f"""
        New contact form submission received!
        
        Name: {name}
        Email: {email}
        Message: {message}
        
        Submitted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        ---
        This email was sent from your PlantGuard AI application.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to SMTP server
        server = smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"])
        server.starttls()
        server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["sender_password"])
        
        # Send email
        text = msg.as_string()
        server.sendmail(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["recipient_email"], text)
        server.quit()
        
        return True
        
    except Exception as e:
        print(f"Email sending error: {e}")
        return False

@app.post("/contact")
async def submit_contact(contact: ContactForm):
    """Submit contact form"""
    try:
        # Log the submission
        print(f"Contact form submitted:")
        print(f"Name: {contact.name}")
        print(f"Email: {contact.email}")
        print(f"Message: {contact.message}")
        
        # Send email notification
        email_sent = send_email(contact.name, contact.email, contact.message)
        
        if email_sent:
            return {
                "success": True,
                "message": "Thank you for your message! We'll get back to you within 24 hours.",
                "submitted_at": datetime.now().isoformat()
            }
        else:
            # Still return success but log the email failure
            print("Warning: Contact form submitted but email notification failed")
            return {
                "success": True,
                "message": "Thank you for your message! We'll get back to you within 24 hours.",
                "submitted_at": datetime.now().isoformat()
            }
        
    except Exception as e:
        print(f"Error processing contact form: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit contact form")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
