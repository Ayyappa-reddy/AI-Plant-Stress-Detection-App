# ============================================================================
# 🌱 AI PLANT DISEASE DETECTION - GOOGLE COLAB TRAINING SCRIPT
# ============================================================================
# Copy and paste this code piece by piece into Google Colab
# Make sure to upload your dataset to Colab first!

# ============================================================================
# PART 1: INSTALL DEPENDENCIES
# ============================================================================
# Run this cell first to install required packages

!pip install torch torchvision fastapi uvicorn python-multipart pillow numpy scikit-learn matplotlib seaborn tqdm albumentations opencv-python

# ============================================================================
# PART 2: IMPORT LIBRARIES
# ============================================================================
# Run this cell to import all necessary libraries

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import zipfile
from google.colab import files

# ============================================================================
# PART 3: UPLOAD YOUR DATASET
# ============================================================================
# Run this cell and upload your plant dataset zip file
# Make sure your dataset has the same structure as described in the README

print("📁 Please upload your plant dataset zip file...")
uploaded = files.upload()

# Extract the dataset
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        print(f"📦 Extracting {filename}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("✅ Dataset extracted successfully!")

# ============================================================================
# PART 4: DATASET CLASS AND FUNCTIONS
# ============================================================================
# Run this cell to define the dataset handling classes

class PlantDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms():
    """Get data augmentation transforms for training and validation"""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def load_dataset(data_dir):
    """Load dataset from directory structure"""
    
    image_paths = []
    labels = []
    class_names = []
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_dirs.sort()  # Sort for consistent ordering
    
    print(f"Found {len(class_dirs)} classes:")
    
    for class_idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(data_dir, class_name)
        class_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"  {class_name}: {len(class_images)} images")
        
        for image_name in class_images:
            image_path = os.path.join(class_path, image_name)
            image_paths.append(image_path)
            labels.append(class_idx)
        
        class_names.append(class_name)
    
    return image_paths, labels, class_names

def create_data_loaders(data_dir, batch_size=32, train_split=0.7, val_split=0.15):
    """Create train, validation, and test data loaders"""
    
    # Load dataset
    image_paths, labels, class_names = load_dataset(data_dir)
    
    # Split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=1-train_split, stratify=labels, random_state=42
    )
    
    val_size = val_split / (train_split + val_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(X_train)} images")
    print(f"  Validation: {len(X_val)} images")
    print(f"  Testing: {len(X_test)} images")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = PlantDataset(X_train, y_train, transform=train_transform)
    val_dataset = PlantDataset(X_val, y_val, transform=val_transform)
    test_dataset = PlantDataset(X_test, y_test, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader, class_names

# ============================================================================
# PART 5: MODEL ARCHITECTURE
# ============================================================================
# Run this cell to define the model architecture

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

# ============================================================================
# PART 6: TRAINING FUNCTIONS
# ============================================================================
# Run this cell to define training and validation functions

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# PART 7: MAIN TRAINING LOOP
# ============================================================================
# Run this cell to start the training process

def main():
    # Configuration
    data_dir = "Data_Set/PlantVillage"  # Adjust this path based on your dataset location
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"Data directory: {data_dir}")
    
    # Load data
    print("Loading dataset...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir, batch_size=batch_size
    )
    
    # Save class names
    class_names_data = json.dumps(class_names, indent=2)
    with open("class_names.json", 'w') as f:
        f.write(class_names_data)
    
    print(f"Class names saved to: class_names.json")
    
    # Initialize model
    num_classes = len(class_names)
    model = PlantDiseaseModel(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    print(f"Model initialized with {num_classes} classes")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training history
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    best_val_acc = 0.0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = "plant_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'num_classes': num_classes
            }, best_model_path)
            print(f"New best model saved! Validation Accuracy: {val_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    # Test on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Save final model
    final_model_path = "plant_model_final.pth"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'class_names': class_names,
        'num_classes': num_classes
    }, final_model_path)
    
    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Model ready for download!")
    
    # Download the model files
    print("\n📥 Downloading model files...")
    files.download("plant_model.pth")
    files.download("class_names.json")
    print("✅ Model files downloaded successfully!")

# ============================================================================
# PART 8: START TRAINING
# ============================================================================
# Run this cell to start the training process

if __name__ == "__main__":
    main()

# ============================================================================
# INSTRUCTIONS FOR USE:
# ============================================================================
# 1. Upload your plant dataset zip file when prompted
# 2. Make sure your dataset has the structure:
#    Data_Set/PlantVillage/
#    ├── Pepper__bell___Bacterial_spot/
#    ├── Pepper__bell___healthy/
#    ├── Potato___Early_blight/
#    └── ... (other classes)
# 3. Run all cells in order
# 4. Wait for training to complete (1-2 hours)
# 5. Download the model files
# 6. Move plant_model.pth and class_names.json to your backend/models/ folder
# ============================================================================
