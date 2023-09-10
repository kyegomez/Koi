import torch
from torch import nn
from copy import deepcopy
import numpy as np



# Define a simple task
class SimpleTask:
    def __init__(self):
        self.a = np.random.uniform(-5, 5)
        self.b = np.random.uniform(-5, 5)

    def f(self, x):
        return self.a * x + self.b

    def sample_data(self, n=100):
        x = np.random.uniform(-5, 5, n)
        y = self.f(x)
        return torch.Tensor(x).unsqueeze(-1), torch.Tensor(y).unsqueeze(-1)

    def loss(self, model):
        x, y = self.sample_data()
        y_pred = model(x)
        return ((y - y_pred)**2).mean()

# Define a task distribution
class SimpleTaskDistribution:
    def sample(self):
        return SimpleTask()

# Initialize the model
model = nn.Sequential(
    nn.Linear(1, 1),
)

# Initialize Koi
koi = Koi(model, step_size=0.01, num_steps=10, num_iterations=1000)

# Sample a task and perform a forward pass
task = SimpleTaskDistribution().sample()
x, y = task.sample_data(n=1)
y_pred = koi.model(x)
print(f"Output before training: {y_pred.item()}")

# Train Koi on tasks
tasks = SimpleTaskDistribution()
koi.train(tasks)

# Perform a forward pass after training
y_pred = koi.model(x)
print(f"Output after training: {y_pred.item()}")

# Inference on new data
x_new, y_new = task.sample_data(n=1)
y_pred_new = koi.model(x_new)
print(f"Output on new data: {y_pred_new.item()}")