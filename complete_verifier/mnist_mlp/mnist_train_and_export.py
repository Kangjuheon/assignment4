import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# MNIST_MLP 구조 (mnist_custom_model_data.py와 동일)
class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 데이터 준비
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 모델, 손실함수, 옵티마이저
model = MNIST_MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 모델 저장
torch.save(model.state_dict(), "mnist_mlp.pth")
print("Saved mnist_mlp.pth")

# ONNX로 export
dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model, dummy_input, "mnist_mlp.onnx",
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("Saved mnist_mlp.onnx") 