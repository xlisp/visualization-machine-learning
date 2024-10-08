import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Define a simple quadratic function
def f(x):
    return (x - 3) ** 2 + 2

# 2. Initialize a tensor with a random starting value
x = torch.tensor([10.0], requires_grad=True)

# 3. Set up the Adam optimizer
optimizer = optim.Adam([x], lr=0.1)

# 4. Run the optimization loop and track the loss
loss_values = []
for _ in range(100):
    optimizer.zero_grad()   # Zero the gradients
    loss = f(x)             # Compute the loss
    loss.backward()         # Backpropagate to compute gradients
    optimizer.step()        # Update the parameter using Adam
    
    # Store the current loss value
    loss_values.append(loss.item())

# 5. Plot the loss over iterations
plt.plot(loss_values)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over iterations using Adam optimizer')
plt.grid(True)
plt.show()

