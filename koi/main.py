import torch
from torch import nn
from copy import deepcopy

class Koi:
    def __init__(
        self,
        model,
        step_size,
        num_steps,
        num_iterations
    ):
        """
        init Koi
        model = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )



        koi = Koi(
            model,
            step_size=0.01,
            num_steps=10, 
            num_iterations=1000
        )
        """
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
    
    def forward(self, x):
        return self.model(x)
    
class Sakana:
    """
    # Initialize the model
    model = nn.Sequential(
        nn.Linear(1, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )

    # Create a random torch input
    x = torch.randn(1, 1)

    # Initialize the swarm
    sakana = Sakana(num_koi=10, model=model, step_size=0.01, num_steps=10, num_iterations=1000)

    # Perform a forward pass through the swarm
    outputs = sakana.forward(x)

    # Print the outputs
    for i, output in enumerate(outputs):
        print(f"Output of Koi instance {i}: {output.item()}")
    """

    def __init__(
            self,
            num_koi, 
            model, 
            step_size, 
            num_steps, 
            num_iterations
        ):
        self.kois = [
            Koi(
                deepcopy(model), step_size, num_steps, num_iterations) for _ in range(num_koi)
            ]
        self.best_positions = [koi.model.state_dict() for koi in self.kois]
        self.best_swarm_position = deepcopy(self.best_positions[0])
        self.velocities = [0 for _ in range(num_koi)]

    def update_positions(self):
        for i, koi in enumerate(self.kois):
            for param in koi.model.parameters():
                self.velocities[i] += 0.1 * (
                    self.best_positions[i] - param.data
                ) + 0.1 * (self.best_swarm_position - param.data)

                param.data += self.velocities[i]

    def forward(self, x):
        return [koi.forward(x) for koi in self.kois]
