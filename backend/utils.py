import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Tuple, List, Dict, Any
import io
from datetime import datetime

def preprocess_image_for_inference(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Preprocess image for model inference
    
    Args:
        image: PIL Image object
        target_size: Target size for the image (width, height)
    
    Returns:
        Preprocessed image tensor
    """
    # Resize image
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(np.array(image)).float()
    
    # Normalize to [0, 1] and convert to [C, H, W] format
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.permute(2, 0, 1)
    
    image_tensor = image_tensor / 255.0
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    image_tensor = (image_tensor - mean) / std
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def get_top_predictions(probabilities: torch.Tensor, class_names: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Get top-k predictions from model output
    
    Args:
        probabilities: Softmax probabilities from model
        class_names: List of class names
        top_k: Number of top predictions to return
    
    Returns:
        List of top predictions with class names and confidence scores
    """
    top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    
    predictions = []
    for i in range(top_k):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        
        predictions.append({
            "class": class_names[idx],
            "confidence": round(prob * 100, 2),
            "class_index": idx
        })
    
    return predictions

def format_class_name(class_name: str) -> str:
    """
    Format class name for display
    
    Args:
        class_name: Raw class name from dataset
    
    Returns:
        Formatted class name for display
    """
    # Replace underscores with spaces and capitalize
    formatted = class_name.replace('_', ' ').replace('  ', ' ')
    
    # Handle special cases
    if 'Pepper__bell' in class_name:
        formatted = formatted.replace('Pepper bell', 'Bell Pepper')
    elif 'Potato' in class_name:
        formatted = formatted.replace('Potato', 'Potato')
    elif 'Tomato' in class_name:
        formatted = formatted.replace('Tomato', 'Tomato')
    
    return formatted.title()

def get_disease_severity(confidence: float) -> str:
    """
    Get disease severity based on confidence score
    
    Args:
        confidence: Confidence score (0-100)
    
    Returns:
        Severity level string
    """
    if confidence >= 90:
        return "High"
    elif confidence >= 70:
        return "Medium"
    elif confidence >= 50:
        return "Low"
    else:
        return "Very Low"

def validate_image_file(file_content: bytes, max_size_mb: int = 10) -> bool:
    """
    Validate uploaded image file
    
    Args:
        file_content: File content in bytes
        max_size_mb: Maximum file size in MB
    
    Returns:
        True if valid, False otherwise
    """
    # Check file size
    if len(file_content) > max_size_mb * 1024 * 1024:
        return False
    
    # Check if it's a valid image by trying to open it
    try:
        image = Image.open(io.BytesIO(file_content))
        image.verify()
        return True
    except:
        return False

def create_response_data(predicted_class: str, confidence: float, 
                        top_predictions: List[Dict[str, Any]], 
                        recommendations: Dict[str, str]) -> Dict[str, Any]:
    """
    Create standardized response data
    
    Args:
        predicted_class: Predicted class name
        confidence: Confidence score
        top_predictions: Top k predictions
        recommendations: Disease recommendations
    
    Returns:
        Formatted response data
    """
    return {
        "predicted_class": predicted_class,
        "formatted_class_name": format_class_name(predicted_class),
        "confidence": confidence,
        "severity": get_disease_severity(confidence),
        "top_predictions": top_predictions,
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat()
    }
