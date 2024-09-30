## -----python use pytorch to Log analysis and   visualization, need auto find log pattern by Frequency of feature,  and need visualization the rnn process

# To visualize the process of the Recurrent Neural Network (RNN) training and inference, you can use different tools to monitor training progress, loss curves, and predictions over time. Below are some strategies to visualize the RNN process:

# ### 1. **Visualizing Training Loss:**
#    You can plot the training loss after each epoch to observe how the model is learning over time.

# ### 2. **Visualizing Hidden State Dynamics:**
#    You can capture and plot the hidden states of the RNN at each time step to understand how the model is processing sequences.

# ### 3. **Visualizing Log Predictions Over Time:**
#    You can visualize the predicted log events and compare them with the actual events for pattern recognition.

# Here’s an enhanced version of the RNN model that includes visualization of these aspects:

# ### Visualization with Matplotlib and TensorBoard

# #### Code Example for Visualization:

# ```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

# TensorBoard writer
writer = SummaryWriter('runs/log_analysis_rnn')

# Sample log sequences
logs = [
    ['INFO', 'INFO', 'ERROR', 'WARNING', 'INFO'],
    ['WARNING', 'ERROR', 'INFO', 'ERROR'],
    ['INFO', 'INFO', 'INFO', 'ERROR', 'ERROR']
]

# Step 1: Preprocess log data (encoding logs to numerical values)
encoder = LabelEncoder()
flattened_logs = [item for sublist in logs for item in sublist]
encoder.fit(flattened_logs)

# Convert logs to sequences of numerical tokens
tokenized_logs = [encoder.transform(log).tolist() for log in logs]
vocab_size = len(encoder.classes_)

# Convert to PyTorch dataset
class LogDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.tensor(sequence[:-1]), torch.tensor(sequence[1:])  # Input: log[:-1], Target: log[1:]

dataset = LogDataset(tokenized_logs)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 2: Define the RNN model
class LogRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(LogRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

# Model Hyperparameters
input_size = vocab_size
hidden_size = 64
output_size = vocab_size
n_layers = 1

# Initialize model, loss, and optimizer
model = LogRNN(input_size, hidden_size, output_size, n_layers)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 3: Training with visualization
def train_rnn(model, dataloader, epochs=50):
    model.train()
    all_losses = []

    for epoch in range(epochs):
        total_loss = 0
        hidden_states = []

        for inputs, targets in dataloader:
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size)

            # Forward pass
            outputs, hidden = model(inputs, hidden)
            hidden_states.append(hidden.detach().numpy())  # Capture hidden state

            loss = loss_fn(outputs.view(-1, output_size), targets.view(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        all_losses.append(avg_loss)

        # Log the loss and hidden states to TensorBoard
        writer.add_scalar('Training Loss', avg_loss, epoch)

        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

    # Step 4: Plot loss curve
    plt.plot(all_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('RNN Training Loss')
    plt.legend()
    plt.show()

# Train the model with visualization
train_rnn(model, dataloader)

# Step 5: Visualization of hidden states
def visualize_hidden_states(hidden_states):
    for i, state in enumerate(hidden_states):
        plt.plot(state[0][0], label=f'Hidden State {i}')
    plt.xlabel('Hidden State Units')
    plt.ylabel('Activation')
    plt.title('Hidden State Dynamics')
    plt.legend()
    plt.show()

# Example prediction for log sequence visualization
def predict(model, input_sequence, future_steps=5):
    model.eval()
    input_sequence = torch.tensor([encoder.transform(input_sequence)], dtype=torch.long)
    hidden = model.init_hidden(input_sequence.size(0))

    predictions = []
    hidden_states = []

    with torch.no_grad():
        for _ in range(future_steps):
            output, hidden = model(input_sequence, hidden)
            predicted_token = torch.argmax(output[:, -1, :], dim=-1)
            predictions.append(predicted_token.item())
            hidden_states.append(hidden.squeeze(0).numpy())
            input_sequence = torch.cat([input_sequence, predicted_token.unsqueeze(0)], dim=1)

    # Visualize hidden states over the prediction
    visualize_hidden_states(hidden_states)

    return encoder.inverse_transform(predictions)

# Example: Visualize hidden state dynamics during prediction
log_sequence = ['INFO', 'INFO', 'ERROR']  # Input sequence
predicted_next_logs = predict(model, log_sequence, future_steps=3)
print(f"Next predicted logs: {predicted_next_logs}")

# Close TensorBoard writer
writer.close()
# ```

# ### Key Visualization Components:

# 1. **Training Loss Curve:**
#    - The training loss is logged using TensorBoard (`SummaryWriter`) and visualized using Matplotlib.
#    - You can also view the training progress live by running `tensorboard --logdir=runs/log_analysis_rnn` in the terminal and opening the TensorBoard dashboard in your browser.

# 2. **Hidden State Dynamics:**
#    - The hidden states are captured during each prediction step and plotted to visualize how they evolve during the sequence processing.
#    - `visualize_hidden_states` plots the activation of the hidden state over time, giving insight into how the RNN’s memory works.

# 3. **Prediction Visualization:**
#    - The predicted log sequence is compared with the actual log sequence to understand the model’s prediction capabilities. The hidden states during prediction are also visualized to track how they evolve as the sequence progresses.

# ### TensorBoard Setup:
# - You can visualize the training loss curve in TensorBoard by running the following command in your terminal:
#   ```bash
#   tensorboard --logdir=runs/log_analysis_rnn
#   ```
# - Open your browser and navigate to `http://localhost:6006` to monitor the training process, loss curves, and more.

# This approach provides both a real-time and historical view of the RNN's training and inference, along with insights into the hidden state dynamics and model behavior.
