import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, class_names

if __name__ == "__main__":
    # Test dataset loading
    data_dir = "../Data_Set/PlantVillage"
    train_loader, val_loader, test_loader, class_names = create_data_loaders(data_dir)
    
    print(f"\nClass names: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    
    # Test a batch
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
        break
