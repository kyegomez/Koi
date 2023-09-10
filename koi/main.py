import torch
from torch import nn
from copy import deepcopy


model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
)


class Koi:
    def __init__(
        self,
        model,
        step_size,
        num_steps,
        num_iterations
    ):
        self.model = model
        self.step_size = step_size
        self.num_steps = num_steps
        self.num_iterations = num_iterations

    def sgd_on_task(self, task):
        initial_parameters = deepcopy(self.model.state_dict())
        for _ in range(self.num_steps):
            loss = task.loss(self.model)
            loss.backward()
            with torch.no_grad():
                for param in self.model.parameters():
                    param -= self.step_size * param.grad
                    param.grad.zero_()
        return self.model.state_dict()
    
    def train(self, tasks):
        for _ in range(self.num_iterations):
            task = tasks.sample()
            final_parameters = self.sgd_on_task(task)
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.data += self.step_size * (final_parameters[name] - param.data)
        return self.model
    
#init Koi
koi = Koi(
    model,
    step_size=0.01,
    num_steps=10, 
    num_iterations=1000
)



