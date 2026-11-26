import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Physics: Exponential Decay (dy/dt = -k*y, y(0) = 1)
k = 2.0
y0 = 1.0

# 2. Build a simple Neural Network (1 input: time t -> 1 output: y(t))
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 3. Create training time points from t = 0 to t = 2 seconds
t_physics = torch.linspace(0, 2, 100).view(-1, 1).requires_grad_(True)
t_initial = torch.tensor([[0.0]])

# 4. Train the PINN using ONLY the differential equation
for epoch in range(2001):
    optimizer.zero_grad()
    
    # Predict y(t) and compute its derivative dy/dt via automatic differentiation
    y_pred = model(t_physics)
    dy_dt = torch.autograd.grad(y_pred, t_physics, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
    
    # Physics Loss: dy/dt + k*y should equal 0 everywhere
    physics_loss = torch.mean((dy_dt + k * y_pred) ** 2)
    
    # Boundary Loss: y(0) should equal 1.0
    initial_loss = (model(t_initial) - y0) ** 2
    
    total_loss = physics_loss + initial_loss
    total_loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.6f}")

# 5. Evaluate and compare against the exact analytical solution y(t) = e^(-kt)
t_test = torch.linspace(0, 2, 200).view(-1, 1)
with torch.no_grad():
    y_pinn = model(t_test).numpy()

t_np = t_test.numpy()
y_exact = y0 * np.exp(-k * t_np)

mse = np.mean((y_pinn - y_exact) ** 2)
print(f"\nFinal Mean Squared Error: {mse:.8f}")

# 6. Plot the results
plt.figure(figsize=(8, 5))
plt.plot(t_np, y_exact, label="Exact Analytical Solution ($e^{-2t}$)", linewidth=2.5, color="royalblue")
plt.plot(t_np, y_pinn, "--", label="PINN Prediction", linewidth=2.5, color="crimson")
plt.title("First PINN: Solving $\\frac{dy}{dt} = -2y$ Without Training Data")
plt.xlabel("Time t (seconds)")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("images/stage1_decay.png", dpi=300, bbox_inches="tight")
plt.show()
