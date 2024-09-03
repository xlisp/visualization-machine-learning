# To use the trained model (`mnist_model.pth`) for recognizing handwritten input numbers, you'll need to load the model, prepare the input image, and then perform inference. Below is a step-by-step guide:

### 1. Load the Required Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

### 2. Define the Model Structure

#Ensure the model structure matches the one used during training.

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

# Initialize the model
model = Net()

### 3. Load the Trained Model Weights

model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()  # Set the model to evaluation mode

### 4. Prepare the Handwritten Input Image

#You need to preprocess the handwritten image to match the format of the MNIST dataset (28x28 pixels, grayscale).

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),  # Ensure the image is grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28 pixels
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with the same mean and std as MNIST
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

### 5. Perform Inference

def recognize_digit(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1, keepdim=True)
    return prediction.item()

# Example usage
image_path = 'path_to_your_handwritten_digit_image3.png'
predicted_digit = recognize_digit(image_path)
print(f'Predicted Digit: {predicted_digit}')

### Summary

# 1. **Load the trained model**: Ensure you use the same architecture as during training.
# 2. **Prepare the input image**: The handwritten digit image must be resized to 28x28 pixels and normalized as per the MNIST dataset.
# 3. **Predict the digit**: The model will output a tensor representing the likelihood of each digit (0-9), and you select the one with the highest probability.
#
# With this setup, you can recognize handwritten digits from images using your trained MNIST model.

# ------- run ----
#/Users/emacspy/EmacsPyPro/emacspy-machine-learning/mnist_recognize_torch.py:33: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#  model.load_state_dict(torch.load("mnist_model.pth"))
#Predicted Digit: 1

