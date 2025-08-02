import torch
import torch.nn as nn
from torchvision import datasets, transforms
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# 1. Teacher AI System
class AITeacher:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
        self.model.eval()
    
    def generate_lesson(self, topic):
        inputs = self.tokenizer(
            f"Explain {topic} to an AI student:",
            return_tensors="pt",
            max_length=128,
            truncation=True
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

# 2. Student AI Model
class AIStudent(nn.Module):
    def __init__(self):
        super().__init__()
        # Text processing pathway
        self.text_fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Image processing pathway
        self.vision_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Combined classifier
        self.classifier = nn.Linear(256 + 64*5*5, 10)  # 10 classes for MNIST
        
    def forward(self, text_emb, img):
        # Process text
        text_features = self.text_fc(text_emb)
        
        # Process image
        img_features = self.vision_net(img)
        img_features = img_features.view(img_features.size(0), -1)
        
        # Combine features
        combined = torch.cat([text_features, img_features], dim=1)
        return self.classifier(combined)

# 3. Training System
class AITrainingSystem:
    def __init__(self):
        self.teacher = AITeacher()
        self.student = AIStudent().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Load MNIST data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.train_data = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform)
        self.test_data = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform)
    
    def train(self, epochs=3, batch_size=64):
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=1000)
        
        loss_history = []
        accuracy_history = []
        
        for epoch in range(epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                # Generate lesson based on digit class
                topic = f"recognizing digit {labels[0].item()}"
                text_emb = self.teacher.generate_lesson(topic)
                text_emb = text_emb.expand(images.size(0), -1)
                
                # Training step
                self.optimizer.zero_grad()
                outputs = self.student(text_emb, images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                loss_history.append(loss.item())
                
                # Print progress
                if i % 50 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
            # Evaluate after each epoch
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    dummy_emb = torch.zeros(1, 768).to(DEVICE).expand(images.size(0), -1)
                    outputs = self.student(dummy_emb, images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            accuracy_history.append(accuracy)
            print(f"Epoch {epoch+1} Test Accuracy: {accuracy:.2%}")
        
        # Save training results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(loss_history)
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        
        plt.subplot(1, 2, 2)
        plt.plot(accuracy_history, marker='o')
        plt.title("Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig("training_results.png")
        print("Training complete. Results saved to training_results.png")

# Run the system
if __name__ == "__main__":
    system = AITrainingSystem()
    system.train(epochs=3, batch_size=64)
