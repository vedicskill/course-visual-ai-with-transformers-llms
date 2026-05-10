# Module 2: Recognition

## Overview

Image recognition, also known as image classification, is the task of assigning labels to images based on their content. This module explores how transformer-based models have improved upon traditional convolutional approaches for this fundamental computer vision task.

## Key Concepts

- Image classification fundamentals
- Convolutional Neural Networks (CNNs)
- Transfer learning and fine-tuning
- Evaluation metrics (accuracy, precision, recall)

## Technical Details

### Traditional CNN Approaches

CNNs use convolutional layers to extract hierarchical features:

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### Transformer-Based Classification

Vision Transformers and hybrid approaches:

```python
# Using pre-trained ViT for classification
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=1000
)
```

## Beginner-Friendly Explanation

Image recognition is like teaching a computer to look at a photo and instantly identify what's in it - "That's a cat!", "That's a car!", or "That's a delicious pizza!". Just like you can recognize objects immediately, AI models learn these patterns by studying thousands of labeled example images.

## Applications

- **Photo Organization**: Automatically tagging and sorting photos
- **Quality Control**: Detecting defects in manufacturing
- **Medical Diagnosis**: Classifying medical images
- **Security**: Facial recognition systems
- **Retail**: Product recognition for inventory

## Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Source Code Reference

Path: `../../m1_recognition/`

```python
# Placeholder for recognition implementation
from torchvision.models import resnet50
import torch

# Load pre-trained ResNet model
model = resnet50(pretrained=True)
model.eval()

# Example inference
def classify_image(image_path):
    # Preprocessing and classification logic
    pass
```

## Notebook Reference

Hands-on recognition examples:

- [CNN Classification](../../m1_recognition/notebooks/)
- [Transfer Learning Demo](../../m1_recognition/notebooks/)
- [ViT Classification](../../m1_recognition/notebooks/)

## Dataset Reference

Standard classification datasets:

- **MNIST**: Handwritten digits (60,000 training, 10,000 test)
- **CIFAR-10/100**: Small images with 10/100 classes
- **ImageNet**: Large-scale dataset with 1,000 classes
- **Fashion-MNIST**: Clothing items classification

## API Reference

[Placeholder for API documentation]

Key libraries:
- PyTorch torchvision.models
- TensorFlow Keras applications
- Hugging Face transformers

## Training Pipeline

[Placeholder for training pipeline]

1. **Data Preparation**: Load and preprocess images
2. **Model Selection**: Choose architecture (ResNet, ViT, etc.)
3. **Transfer Learning**: Fine-tune pre-trained weights
4. **Training Loop**: Optimize with backpropagation
5. **Validation**: Monitor performance on held-out data
6. **Testing**: Final evaluation on test set

## Configuration Files

[Placeholder for configuration]

```yaml
# Classification training configuration
model:
  architecture: resnet50
  num_classes: 1000
  pretrained: true

training:
  batch_size: 32
  learning_rate: 1e-3
  epochs: 50
  optimizer: adam

data:
  dataset: imagenet
  image_size: 224
  augmentation: true
```