# use pytorch Implement MNIST

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

### 3. Define Hyperparameters

batch_size = 64
learning_rate = 0.01
epochs = 100

### 4. Prepare the MNIST Dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

### 5. Build the Neural Network Model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

### 6. Initialize the Model, Loss Function, and Optimizer

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

### 7. Train the Model

for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1}/{epochs} [Batch: {batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

### 8. Test the Model

model.eval()
test_loss = 0
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)

print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

### 9. Save the Model (Optional)

torch.save(model.state_dict(), "mnist_model.pth")

### Summary
# This script loads the MNIST dataset, defines a simple neural network model, trains it on the dataset, and evaluates its performance on the test set. The network architecture used here is quite basic, and you could improve it with more layers, dropout, or other advanced techniques for better performance.

## ----- very fast ------ in m3 ------
# @ python MNIST_torch.py
# Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
# Failed to download (trying next):
# HTTP Error 403: Forbidden
# 
# Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
# Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to ./data/MNIST/raw/train-images-idx3-ubyte.gz
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9912422/9912422 [00:01<00:00, 6148112.22it/s]
# Extracting ./data/MNIST/raw/train-images-idx3-ubyte.gz to ./data/MNIST/raw
# 
# Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
# Failed to download (trying next):
# HTTP Error 403: Forbidden
# 
# Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
# Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to ./data/MNIST/raw/train-labels-idx1-ubyte.gz
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 28881/28881 [00:00<00:00, 43471.05it/s]
# Extracting ./data/MNIST/raw/train-labels-idx1-ubyte.gz to ./data/MNIST/raw
# 
# Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
# Failed to download (trying next):
# HTTP Error 403: Forbidden
# 
# Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
# Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw/t10k-images-idx3-ubyte.gz
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1648877/1648877 [00:01<00:00, 1620685.15it/s]
# Extracting ./data/MNIST/raw/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw
# 
# Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
# Failed to download (trying next):
# HTTP Error 403: Forbidden
# 
# Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
# Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4542/4542 [00:00<00:00, 38500.31it/s]
# Extracting ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw
# 
# Epoch: 1/10 [Batch: 0/60000] Loss: 2.285485
# Epoch: 1/10 [Batch: 6400/60000] Loss: 1.787282
# Epoch: 1/10 [Batch: 12800/60000] Loss: 1.077599
# Epoch: 1/10 [Batch: 19200/60000] Loss: 0.655586
# Epoch: 1/10 [Batch: 25600/60000] Loss: 0.473320
# Epoch: 1/10 [Batch: 32000/60000] Loss: 0.526482
# Epoch: 1/10 [Batch: 38400/60000] Loss: 0.373468
# Epoch: 1/10 [Batch: 44800/60000] Loss: 0.517464
# Epoch: 1/10 [Batch: 51200/60000] Loss: 0.446572
# Epoch: 1/10 [Batch: 57600/60000] Loss: 0.406734
# Epoch: 2/10 [Batch: 0/60000] Loss: 0.674305
# Epoch: 2/10 [Batch: 6400/60000] Loss: 0.377594
# Epoch: 2/10 [Batch: 12800/60000] Loss: 0.215526
# Epoch: 2/10 [Batch: 19200/60000] Loss: 0.293928
# Epoch: 2/10 [Batch: 25600/60000] Loss: 0.245266
# Epoch: 2/10 [Batch: 32000/60000] Loss: 0.297042
# Epoch: 2/10 [Batch: 38400/60000] Loss: 0.283664
# Epoch: 2/10 [Batch: 44800/60000] Loss: 0.340037
# Epoch: 2/10 [Batch: 51200/60000] Loss: 0.383973
# Epoch: 2/10 [Batch: 57600/60000] Loss: 0.453223
# Epoch: 3/10 [Batch: 0/60000] Loss: 0.213752
# Epoch: 3/10 [Batch: 6400/60000] Loss: 0.314306
# Epoch: 3/10 [Batch: 12800/60000] Loss: 0.150672
# Epoch: 3/10 [Batch: 19200/60000] Loss: 0.273192
# Epoch: 3/10 [Batch: 25600/60000] Loss: 0.248815
# Epoch: 3/10 [Batch: 32000/60000] Loss: 0.481632
# Epoch: 3/10 [Batch: 38400/60000] Loss: 0.284605
# Epoch: 3/10 [Batch: 44800/60000] Loss: 0.215667
# Epoch: 3/10 [Batch: 51200/60000] Loss: 0.496624
# Epoch: 3/10 [Batch: 57600/60000] Loss: 0.227222
# Epoch: 4/10 [Batch: 0/60000] Loss: 0.260029
# Epoch: 4/10 [Batch: 6400/60000] Loss: 0.215243
# Epoch: 4/10 [Batch: 12800/60000] Loss: 0.171734
# Epoch: 4/10 [Batch: 19200/60000] Loss: 0.211567
# Epoch: 4/10 [Batch: 25600/60000] Loss: 0.272625
# Epoch: 4/10 [Batch: 32000/60000] Loss: 0.345563
# Epoch: 4/10 [Batch: 38400/60000] Loss: 0.251216
# Epoch: 4/10 [Batch: 44800/60000] Loss: 0.210512
# Epoch: 4/10 [Batch: 51200/60000] Loss: 0.218367
# Epoch: 4/10 [Batch: 57600/60000] Loss: 0.270740
# Epoch: 5/10 [Batch: 0/60000] Loss: 0.171540
# Epoch: 5/10 [Batch: 6400/60000] Loss: 0.182449
# Epoch: 5/10 [Batch: 12800/60000] Loss: 0.193923
# Epoch: 5/10 [Batch: 19200/60000] Loss: 0.399605
# Epoch: 5/10 [Batch: 25600/60000] Loss: 0.176479
# Epoch: 5/10 [Batch: 32000/60000] Loss: 0.263907
# Epoch: 5/10 [Batch: 38400/60000] Loss: 0.143796
# Epoch: 5/10 [Batch: 44800/60000] Loss: 0.117058
# Epoch: 5/10 [Batch: 51200/60000] Loss: 0.102552
# Epoch: 5/10 [Batch: 57600/60000] Loss: 0.234998
# Epoch: 6/10 [Batch: 0/60000] Loss: 0.190606
# Epoch: 6/10 [Batch: 6400/60000] Loss: 0.161692
# Epoch: 6/10 [Batch: 12800/60000] Loss: 0.273620
# Epoch: 6/10 [Batch: 19200/60000] Loss: 0.274887
# Epoch: 6/10 [Batch: 25600/60000] Loss: 0.119512
# Epoch: 6/10 [Batch: 32000/60000] Loss: 0.187856
# Epoch: 6/10 [Batch: 38400/60000] Loss: 0.142739
# Epoch: 6/10 [Batch: 44800/60000] Loss: 0.127890
# Epoch: 6/10 [Batch: 51200/60000] Loss: 0.253540
# Epoch: 6/10 [Batch: 57600/60000] Loss: 0.135013
# Epoch: 7/10 [Batch: 0/60000] Loss: 0.139322
# Epoch: 7/10 [Batch: 6400/60000] Loss: 0.217386
# Epoch: 7/10 [Batch: 12800/60000] Loss: 0.127039
# Epoch: 7/10 [Batch: 19200/60000] Loss: 0.138367
# Epoch: 7/10 [Batch: 25600/60000] Loss: 0.167661
# Epoch: 7/10 [Batch: 32000/60000] Loss: 0.103273
# Epoch: 7/10 [Batch: 38400/60000] Loss: 0.158958
# Epoch: 7/10 [Batch: 44800/60000] Loss: 0.232634
# Epoch: 7/10 [Batch: 51200/60000] Loss: 0.269437
# Epoch: 7/10 [Batch: 57600/60000] Loss: 0.234816
# Epoch: 8/10 [Batch: 0/60000] Loss: 0.051134
# Epoch: 8/10 [Batch: 6400/60000] Loss: 0.094156
# Epoch: 8/10 [Batch: 12800/60000] Loss: 0.271425
# Epoch: 8/10 [Batch: 19200/60000] Loss: 0.155359
# Epoch: 8/10 [Batch: 25600/60000] Loss: 0.183958
# Epoch: 8/10 [Batch: 32000/60000] Loss: 0.182390
# Epoch: 8/10 [Batch: 38400/60000] Loss: 0.154530
# Epoch: 8/10 [Batch: 44800/60000] Loss: 0.133099
# Epoch: 8/10 [Batch: 51200/60000] Loss: 0.107830
# Epoch: 8/10 [Batch: 57600/60000] Loss: 0.141398
# Epoch: 9/10 [Batch: 0/60000] Loss: 0.130976
# Epoch: 9/10 [Batch: 6400/60000] Loss: 0.278118
# Epoch: 9/10 [Batch: 12800/60000] Loss: 0.233672
# Epoch: 9/10 [Batch: 19200/60000] Loss: 0.132633
# Epoch: 9/10 [Batch: 25600/60000] Loss: 0.118954
# Epoch: 9/10 [Batch: 32000/60000] Loss: 0.127910
# Epoch: 9/10 [Batch: 38400/60000] Loss: 0.176615
# Epoch: 9/10 [Batch: 44800/60000] Loss: 0.099357
# Epoch: 9/10 [Batch: 51200/60000] Loss: 0.089455
# Epoch: 9/10 [Batch: 57600/60000] Loss: 0.162767
# Epoch: 10/10 [Batch: 0/60000] Loss: 0.028447
# Epoch: 10/10 [Batch: 6400/60000] Loss: 0.058944
# Epoch: 10/10 [Batch: 12800/60000] Loss: 0.173397
# Epoch: 10/10 [Batch: 19200/60000] Loss: 0.111585
# Epoch: 10/10 [Batch: 25600/60000] Loss: 0.079948
# Epoch: 10/10 [Batch: 32000/60000] Loss: 0.082440
# Epoch: 10/10 [Batch: 38400/60000] Loss: 0.110963
# Epoch: 10/10 [Batch: 44800/60000] Loss: 0.058552
# Epoch: 10/10 [Batch: 51200/60000] Loss: 0.152358
# Epoch: 10/10 [Batch: 57600/60000] Loss: 0.090184
# Test set: Average loss: 0.0019, Accuracy: 9640/10000 (96.40%)
# 
