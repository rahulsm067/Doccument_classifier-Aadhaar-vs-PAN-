import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
import os
from tqdm import tqdm
from src.logger import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Config

train_dir = "data/train"
val_dir = "data/val"
batch_size = 16
num_epochs = 30
num_classes = 3  # Aadhaar, PAN, Other
lr = 1e-4
weight_decay = 1e-4
patience = 5   # Early stopping patience
grad_clip = 2.0  # Max norm for gradient clipping

# Data augmentation

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Early stopping utility

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_acc = 0
        self.early_stop = False

    def step(self, acc):
        if acc > self.best_acc:
            self.best_acc = acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True



# Model setup (ResNet18)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),  # dropout for regularization
    nn.Linear(num_features, num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)


# Training loop

best_acc = 0.0
os.makedirs("models", exist_ok=True)
early_stopper = EarlyStopping(patience=patience)

# Lists to store accuracies
train_acc_history = []
val_acc_history = []

logger.info("Starting CNN training...")

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    val_loss /= len(val_loader)

    # Save history
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    logger.info(f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # LR scheduling
    scheduler.step(val_acc)

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "models/cnn_model.pth")
        logger.info(f"Validation accuracy improved to {best_acc:.2f}%. Saving best model.")

    # Early stopping
    early_stopper.step(val_acc)
    if early_stopper.early_stop:
        logger.info("Early stopping triggered. Training stopped.")
        break

# ==== After training ====
avg_train_acc = sum(train_acc_history) / len(train_acc_history)
avg_val_acc = sum(val_acc_history) / len(val_acc_history)

logger.info(f"Average Train Accuracy: {avg_train_acc:.2f}%")
logger.info(f"Average Validation Accuracy: {avg_val_acc:.2f}%")
logger.info(f"Best Validation Accuracy: {best_acc:.2f}%")

logger.info("Training finished.")
