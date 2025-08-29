import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from dataset import create_data_loaders

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

def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Configuration
    data_dir = "../Data_Set/PlantVillage"
    model_save_dir = "../backend/models"
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Data directory: {data_dir}")
    
    # Create model save directory
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir, batch_size=batch_size
    )
    
    # Save class names for inference
    class_names_path = os.path.join(model_save_dir, "class_names.json")
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    
    print(f"Class names saved to: {class_names_path}")
    
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
            best_model_path = os.path.join(model_save_dir, "plant_model.pth")
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
    plot_path = os.path.join(model_save_dir, "training_history.png")
    plot_training_history(train_losses, train_accs, val_losses, val_accs, plot_path)
    print(f"Training history plot saved to: {plot_path}")
    
    # Test on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Save final model
    final_model_path = os.path.join(model_save_dir, "plant_model_final.pth")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'class_names': class_names,
        'num_classes': num_classes
    }, final_model_path)
    
    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Model ready for inference!")

if __name__ == "__main__":
    main()
