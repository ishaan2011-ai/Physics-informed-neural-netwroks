import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Physics: Damped Mass-Spring System
# Equation: m*(d^2x/dt^2) + c*(dx/dt) + k*x = 0
m = 1.0   # mass (kg)
c = 1.0   # friction / damping coefficient
k = 25.0  # spring stiffness

# 2. Neural Network 
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 3. Training points from t = 0 to t = 3 seconds
t_physics = torch.linspace(0, 3, 250).view(-1, 1).requires_grad_(True)
t_initial = torch.tensor([[0.0]], requires_grad=True)

# 4. Train the PINN
for epoch in range(5001):
    optimizer.zero_grad()
    
    # Predict position x(t)
    x_pred = model(t_physics)
    
    # 1st Derivative: Velocity (dx/dt)
    dx_dt = torch.autograd.grad(
        x_pred, t_physics, grad_outputs=torch.ones_like(x_pred), create_graph=True
    )[0]
    
    # 2nd Derivative: Acceleration (d^2x/dt^2) — derivative of velocity!
    d2x_dt2 = torch.autograd.grad(
        dx_dt, t_physics, grad_outputs=torch.ones_like(dx_dt), create_graph=True
    )[0]
    
    # Physics Loss: Newton's 2nd Law residual must equal 0
    physics_loss = torch.mean((m * d2x_dt2 + c * dx_dt + k * x_pred) ** 2)
    
    # Initial Conditions at t = 0:
    # Rule A: Initial position x(0) = 1.0
    x0_pred = model(t_initial)
    loss_x0 = (x0_pred - 1.0) ** 2
    
    # Rule B: Initial velocity dx/dt(0) = 0.0 (released from rest)
    v0_pred = torch.autograd.grad(
        x0_pred, t_initial, grad_outputs=torch.ones_like(x0_pred), create_graph=True
    )[0]
    loss_v0 = (v0_pred - 0.0) ** 2
    
    # Total Loss (giving extra weight to initial conditions so the wave anchors cleanly)
    total_loss = physics_loss + 10.0 * (loss_x0 + loss_v0)
    total_loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.6f}")

# 5. Compare against the exact textbook physics solution
t_test = torch.linspace(0, 3, 300).view(-1, 1)
with torch.no_grad():
    x_pinn = model(t_test).numpy()

t_np = t_test.numpy()
gamma = c / (2 * m)
omega_0 = np.sqrt(k / m)

if gamma < omega_0:
    # Underdamped: bounces with shrinking amplitude
    omega_d = np.sqrt(omega_0**2 - gamma**2)
    x_exact = np.exp(-gamma * t_np) * (np.cos(omega_d * t_np) + (gamma / omega_d) * np.sin(omega_d * t_np))
elif np.isclose(gamma, omega_0):
    # Critically damped: fastest return to zero without bouncing!
    x_exact = (1.0 + gamma * t_np) * np.exp(-gamma * t_np)
else:
    # Overdamped: slow ooze through thick honey
    alpha = np.sqrt(gamma**2 - omega_0**2)
    r1, r2 = -gamma + alpha, -gamma - alpha
    c1 = r2 / (r2 - r1)
    c2 = -r1 / (r2 - r1)
    x_exact = c1 * np.exp(r1 * t_np) + c2 * np.exp(r2 * t_np)
mse = np.mean((x_pinn - x_exact) ** 2)
print(f"\nFinal Mean Squared Error: {mse:.8f}")

# 6. Plot the bouncing spring wave
plt.figure(figsize=(9, 5))
plt.plot(t_np, x_exact, label="Exact Physics Solution", linewidth=2.5, color="royalblue")
plt.plot(t_np, x_pinn, "--", label="Stage 2 PINN Prediction", linewidth=2.5, color="crimson")
plt.title("Stage 2 PINN: Damped Harmonic Oscillator ($m\\ddot{x} + c\\dot{x} + kx = 0$)")
plt.xlabel("Time t (seconds)")
plt.ylabel("Displacement x(t) [meters]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("images/stage2_oscillator.png", dpi=300, bbox_inches="tight")
plt.show()
