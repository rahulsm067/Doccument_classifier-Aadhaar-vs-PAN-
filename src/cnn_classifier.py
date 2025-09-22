import time
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from src.logger import logger


class CNNClassifier:
    def __init__(self, model_path="models/cnn_model.pth", num_classes=3, device=None):
        """
        Initialize CNN model for inference.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained ResNet18 and modify for classification (MUST match training definition)
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),  # must match train_cnn.py
            nn.Linear(self.model.fc.in_features, num_classes)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.classes = ["AADHAAR", "PAN", "OTHER"]

    def predict(self, image_path):
        """
        Run CNN prediction on a document image.
        Returns: (label, confidence, inference_time)
        """
        start_time = time.time()

        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)

        inference_time = time.time() - start_time
        label = self.classes[pred_idx.item()]
        confidence = conf.item()

        logger.info(f"Predicted {label} with confidence {confidence:.2f}")

        return label, confidence, inference_time
