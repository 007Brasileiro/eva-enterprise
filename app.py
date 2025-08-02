import torch
import torch.nn as nn
from torchvision import datasets, transforms
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Configuração para Windows
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# 1. Sistema Professor IA
class AITeacher:
    def __init__(self):
        print("Carregando modelo BERT...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
        self.model.eval()
        print("BERT carregado com sucesso!")
    
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

# 2. Modelo Estudante IA (CORRIGIDO)
class AIStudent(nn.Module):
    def __init__(self):
        super().__init__()
        # Camada de processamento de texto
        self.text_fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )  # PARÊNTESE FECHADO AQUI
        
        # Camada de processamento de imagem
        self.vision_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  # PARÊNTESE FECHADO AQUI
        
        # Camada de classificação
        self.classifier = nn.Linear(256 + 64*7*7, 10)
        
    def forward(self, text_emb, img):
        # Processamento de texto
        text_out = self.text_fc(text_emb)
        
        # Processamento de imagem
        img_out = self.vision_net(img)
        img_out = img_out.view(img_out.size(0), -1)
        
        # Combinação
        combined = torch.cat([text_out, img_out], dim=1)
        return self.classifier(combined)

# 3. Sistema de Treinamento
class AITrainingSystem:
    def __init__(self):
        self.teacher = AITeacher()
        self.student = AIStudent().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Carrega MNIST
        print("Carregando dataset MNIST...")
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
        print("MNIST carregado com sucesso!")
    
    def train(self, epochs=3, batch_size=64):
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=1000)
        
        print("\nIniciando treinamento...")
        for epoch in range(epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                # Gera lição
                topic = f"dígito {labels[0].item()}"
                text_emb = self.teacher.generate_lesson(topic)
                text_emb = text_emb.expand(images.size(0), -1)
                
                # Passo de treino
                self.optimizer.zero_grad()
                outputs = self.student(text_emb, images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                # Progresso
                if i % 50 == 0:
                    print(f"Época {epoch+1}/{epochs} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
            # Avaliação
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
            
            print(f"Época {epoch+1} | Acurácia: {correct/total:.2%}")

# Execução
if __name__ == "__main__":
    system = AITrainingSystem()
    system.train(epochs=2, batch_size=64)
