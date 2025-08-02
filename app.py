import torch
import torch.nn as nn
from torchvision import datasets, transforms
from transformers import BertModel, BertTokenizer
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# 1. Simplified Data Loading
@st.cache_resource
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    return train_data

# 2. Robust Teacher System
class AITeacher:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
        self.model.eval()
    
    def generate_lesson(self, topic):
        inputs = self.tokenizer(
            f"Explain {topic} to an AI student with examples:",
            return_tensors="pt",
            max_length=128,
            truncation=True
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

# 3. Fixed Student Model
class AIStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.nlp_layer = nn.Linear(768, 64)
        self.vision_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 5 * 5 + 64, 10)  # MNIST has 10 classes
        
    def forward(self, nlp_emb, img):
        # Process NLP input
        nlp_out = torch.relu(self.nlp_layer(nlp_emb))
        
        # Process image input
        img_out = self.vision_conv(img)
        img_out = img_out.view(img_out.size(0), -1)
        
        # Combine features
        combined = torch.cat([nlp_out, img_out], dim=1)
        return self.fc(combined)

# 4. Streamlit App
def main():
    st.set_page_config(layout="wide")
    st.title("🏢 EVA Enterprise - AI Teaching System")
    
    # Initialize system
    if 'system' not in st.session_state:
        st.session_state.teacher = AITeacher()
        st.session_state.student = AIStudent().to(DEVICE)
        st.session_state.optimizer = torch.optim.Adam(st.session_state.student.parameters(), lr=0.001)
        st.session_state.loss_fn = nn.CrossEntropyLoss()
        st.session_state.train_data = load_data()
        st.session_state.loss_history = []
    
    if st.button("Start Training Session"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simple training loop
        train_loader = DataLoader(st.session_state.train_data, batch_size=64, shuffle=True)
        
        for epoch in range(3):  # 3 epochs
            for i, (images, labels) in enumerate(train_loader):
                if i > 30:  # Limit for demo
                    break
                
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                # Generate lesson based on label
                topic = f"number {labels[0].item()}"
                nlp_emb = st.session_state.teacher.generate_lesson(topic)
                
                # Expand NLP embedding to match batch size
                nlp_emb = nlp_emb.expand(images.size(0), -1)
                
                # Training step
                st.session_state.optimizer.zero_grad()
                outputs = st.session_state.student(nlp_emb, images)
                loss = st.session_state.loss_fn(outputs, labels)
                loss.backward()
                st.session_state.optimizer.step()
                
                st.session_state.loss_history.append(loss.item())
                
                # Update UI
                progress = (epoch * len(train_loader) + i) / (3 * len(train_loader))
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")
        
        # Show results
        st.success("Training Complete!")
        fig, ax = plt.subplots()
        ax.plot(st.session_state.loss_history)
        ax.set_title("Training Loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
