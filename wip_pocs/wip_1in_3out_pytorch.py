## pytorch how can like keras function api : one input picture output predict age , name , man/woman.

# In PyTorch, replicating the Keras Functional API for multiple outputs (age, name, and gender) from a single input (picture) involves creating a custom neural network class that defines a shared backbone (for feature extraction) and multiple output heads (for predicting different tasks). Here's a step-by-step guide on how to achieve this:

# ### Step 1: Import Libraries
# ```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# ```

# ### Step 2: Define the Model
# You'll define a model with a shared backbone (usually a convolutional layer for image feature extraction), followed by three separate branches for the three outputs: age, name, and gender.

# ```python
class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()

        # Shared Backbone (Convolutional layers for image processing)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 3 for RGB channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers after flattening
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),  # assuming input size (3, 64, 64)
            nn.ReLU()
        )

        # Age Prediction Head (Regression output)
        self.age_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Single output for age prediction
        )

        # Name Prediction Head (Classification output with softmax)
        self.name_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_names),  # Output size = number of possible names (num_names)
            nn.Softmax(dim=1)  # Output a probability distribution over possible names
        )

        # Gender Prediction Head (Binary classification output)
        self.gender_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2),  # Output 2 units (Man or Woman)
            nn.Softmax(dim=1)   # Output a probability distribution (Man/Woman)
        )

    def forward(self, x):
        # Shared Backbone
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flattening the output for FC layers
        x = self.fc(x)

        # Predict age, name, and gender
        age = self.age_head(x)
        name = self.name_head(x)
        gender = self.gender_head(x)

        return age, name, gender
# ```

# ### Step 3: Instantiate the Model
# ```python
num_names = 100  # Assuming 100 possible names for the name classification task
model = MultiTaskModel()
# ```

# ### Step 4: Define Loss Functions and Optimizer
# Each output will have its own loss function. For example:
# - Age prediction (a regression task) might use `MSELoss`.
# - Name prediction (a classification task) might use `CrossEntropyLoss`.
# - Gender prediction (a binary classification task) might also use `CrossEntropyLoss`.

# ```python
# Loss functions
criterion_age = nn.MSELoss()  # Regression for age
criterion_name = nn.CrossEntropyLoss()  # Classification for name
criterion_gender = nn.CrossEntropyLoss()  # Binary classification for gender

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# ```

# ### Step 5: Training Loop
# During training, you will calculate each loss separately, then combine them to backpropagate the error through the shared backbone and the task-specific heads.

# ```python
for epoch in range(num_epochs):
    for images, age_labels, name_labels, gender_labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        age_pred, name_pred, gender_pred = model(images)

        # Calculate losses
        loss_age = criterion_age(age_pred, age_labels)
        loss_name = criterion_name(name_pred, name_labels)
        loss_gender = criterion_gender(gender_pred, gender_labels)

        # Total loss (you can weigh these losses if necessary)
        total_loss = loss_age + loss_name + loss_gender

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}')
# ```

# ### Step 6: Inference
# During inference, after passing an image through the model, you will receive three outputs: predicted age, name, and gender.

# ```python
with torch.no_grad():
    model.eval()
    image = ...  # Load a single image
    age_pred, name_pred, gender_pred = model(image.unsqueeze(0))  # Add batch dimension

    # Decode predictions
    predicted_age = age_pred.item()
    predicted_name = torch.argmax(name_pred, dim=1)
    predicted_gender = torch.argmax(gender_pred, dim=1)

    print(f'Predicted Age: {predicted_age}')
    print(f'Predicted Name ID: {predicted_name.item()}')
    print(f'Predicted Gender: {"Man" if predicted_gender.item() == 0 else "Woman"}')
# ```

# ### Customization Options:
# - You can modify the backbone architecture to be deeper (ResNet, VGG, etc.) if the image size is larger or the task is more complex.
# - Add dropout or batch normalization to improve regularization.
# - You can adjust the loss function to give more importance to certain tasks (e.g., by weighting the losses differently).

# This setup mimics the flexibility of Keras' Functional API but using PyTorch's more manual, yet highly customizable, approach.
