import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Physics Parameter
alpha = 0.1

# 2. Neural Network: 3 Inputs (x, y, t) -> 1 Output (Temperature T)
model = nn.Sequential(
    nn.Linear(3, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# 3. Sample Collocation Points (Detached leaf tensors)
x_phys = torch.rand(2000, 1, requires_grad=True)
y_phys = torch.rand(2000, 1, requires_grad=True)
t_phys = (torch.rand(2000, 1) * 0.5).detach().requires_grad_(True)

# Initial condition points at t = 0: T(x, y, 0) = sin(pi*x)*sin(pi*y)
x_init = torch.rand(500, 1)
y_init = torch.rand(500, 1)
t_init = torch.zeros(500, 1)
xyt_init = torch.cat((x_init, y_init, t_init), dim=1).detach()
T_init_exact = (torch.sin(np.pi * x_init) * torch.sin(np.pi * y_init)).detach()

# Boundary points along all 4 cold edges (T = 0)
n_bc = 400
r = torch.rand(n_bc, 1)
t_b = torch.rand(n_bc, 1) * 0.5
z = torch.zeros(n_bc, 1)
o = torch.ones(n_bc, 1)

xyt_bc = torch.cat((
    torch.cat((z, r, t_b), dim=1),
    torch.cat((o, r, t_b), dim=1),
    torch.cat((r, z, t_b), dim=1),
    torch.cat((r, o, t_b), dim=1),
), dim=0).detach()

# 4. Training Loop
for epoch in range(3001):
    optimizer.zero_grad()

    xyt_phys = torch.cat((x_phys, y_phys, t_phys), dim=1)
    T_pred = model(xyt_phys)
    ones = torch.ones_like(T_pred)

    dT_dt, = torch.autograd.grad(T_pred, t_phys, grad_outputs=ones, create_graph=True)
    dT_dx, = torch.autograd.grad(T_pred, x_phys, grad_outputs=ones, create_graph=True)
    dT_dy, = torch.autograd.grad(T_pred, y_phys, grad_outputs=ones, create_graph=True)

    d2T_dx2, = torch.autograd.grad(dT_dx, x_phys, grad_outputs=ones, create_graph=True)
    d2T_dy2, = torch.autograd.grad(dT_dy, y_phys, grad_outputs=ones, create_graph=True)

    physics_loss = torch.mean((dT_dt - alpha * (d2T_dx2 + d2T_dy2)) ** 2)
    init_loss = torch.mean((model(xyt_init) - T_init_exact) ** 2)
    bc_loss = torch.mean(model(xyt_bc) ** 2)

    total_loss = physics_loss + 10.0 * init_loss + 10.0 * bc_loss
    total_loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.6f}")

# 5. Evaluate at t = 0.4s and Save/Plot Heatmaps
grid_n = 60
x_lin = np.linspace(0, 1, grid_n)
y_lin = np.linspace(0, 1, grid_n)
X, Y = np.meshgrid(x_lin, y_lin)
t_eval = 0.4

test_pts = torch.tensor(
    np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, t_eval))),
    dtype=torch.float32
)

with torch.no_grad():
    T_pinn_grid = model(test_pts).numpy().reshape(grid_n, grid_n)

T_exact_grid = np.exp(-2 * (np.pi**2) * alpha * t_eval) * np.sin(np.pi * X) * np.sin(np.pi * Y)
mse = np.mean((T_pinn_grid - T_exact_grid) ** 2)
print(f"\n2D Plate Mean Squared Error at t = {t_eval}s: {mse:.8f}")

#6. Plot the heatmaps  
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4.2))

im0 = ax0.contourf(X, Y, T_exact_grid, levels=30, cmap="inferno", vmin=0, vmax=0.5)
ax0.set_title(f"Exact Physics (t = {t_eval}s)")
plt.colorbar(im0, ax=ax0)

im1 = ax1.contourf(X, Y, T_pinn_grid, levels=30, cmap="inferno", vmin=0, vmax=0.5)
ax1.set_title(f"Stage 3 PINN Prediction (t = {t_eval}s)")
plt.colorbar(im1, ax=ax1)

error_grid = np.abs(T_exact_grid - T_pinn_grid)
im2 = ax2.contourf(X, Y, error_grid, levels=30, cmap="viridis")
ax2.set_title("Absolute Error |Exact - PINN|")
plt.colorbar(im2, ax=ax2)

for ax in (ax0, ax1, ax2):
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")

plt.tight_layout()
plt.savefig("images/stage3_heat_2d.png", dpi=300, bbox_inches="tight")
plt.show()

