import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Fluid Properties: Kinematic Viscosity (nu) & Pipe Dimensions
nu = 0.05      # Viscosity (higher = thicker fluid like oil, lower = runny like water)
L_x = 2.0      # Pipe length (0 to 2m)
L_y = 1.0      # Pipe height (0 to 1m)

# 2. Neural Network: 2 Inputs (x, y) -> 3 Outputs (u_vel, v_vel, pressure)
model = nn.Sequential(
    nn.Linear(2, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 3)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# 3. Sample Collocation Points Inside the Pipe
N_phys = 3000
x_phys = (torch.rand(N_phys, 1) * L_x).requires_grad_(True)
y_phys = (torch.rand(N_phys, 1) * L_y).requires_grad_(True)

# Boundary Conditions:
N_bc = 400
# Top & Bottom Walls (y = 0 and y = 1): No-slip condition (u = 0, v = 0)
x_wall = torch.rand(N_bc, 1) * L_x
y_bot = torch.zeros(N_bc, 1)
y_top = torch.ones(N_bc, 1) * L_y

# Inlet (x = 0): Parabolic inflow profile u(0, y) = 4 * y * (1 - y), v = 0
y_in = torch.rand(N_bc, 1) * L_y
x_in = torch.zeros(N_bc, 1)
u_in_exact = 4.0 * y_in * (1.0 - y_in)

# Outlet (x = 2): Zero pressure exit (p = 0)
y_out = torch.rand(N_bc, 1) * L_y
x_out = torch.ones(N_bc, 1) * L_x

# 4. Training Loop: Solving 3 Coupled Navier-Stokes PDEs
for epoch in range(4001):
    optimizer.zero_grad()

    xy_phys = torch.cat((x_phys, y_phys), dim=1)
    out = model(xy_phys)
    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]
    ones = torch.ones_like(u)

    # 1st Partial Derivatives
    u_x, = torch.autograd.grad(u, x_phys, grad_outputs=ones, create_graph=True)
    u_y, = torch.autograd.grad(u, y_phys, grad_outputs=ones, create_graph=True)
    v_x, = torch.autograd.grad(v, x_phys, grad_outputs=ones, create_graph=True)
    v_y, = torch.autograd.grad(v, y_phys, grad_outputs=ones, create_graph=True)
    p_x, = torch.autograd.grad(p, x_phys, grad_outputs=ones, create_graph=True)
    p_y, = torch.autograd.grad(p, y_phys, grad_outputs=ones, create_graph=True)

    # 2nd Partial Derivatives (Viscous friction diffusion)
    u_xx, = torch.autograd.grad(u_x, x_phys, grad_outputs=ones, create_graph=True)
    u_yy, = torch.autograd.grad(u_y, y_phys, grad_outputs=ones, create_graph=True)
    v_xx, = torch.autograd.grad(v_x, x_phys, grad_outputs=ones, create_graph=True)
    v_yy, = torch.autograd.grad(v_y, y_phys, grad_outputs=ones, create_graph=True)

    # Equation 1: Conservation of Mass (Incompressible fluid)
    f_mass = u_x + v_y

    # Equation 2: X-Momentum (Navier-Stokes)
    f_mom_x = (u * u_x + v * u_y) + p_x - nu * (u_xx + u_yy)

    # Equation 3: Y-Momentum (Navier-Stokes)
    f_mom_y = (u * v_x + v * v_y) + p_y - nu * (v_xx + v_yy)

    physics_loss = torch.mean(f_mass**2 + f_mom_x**2 + f_mom_y**2)

    # Boundary Losses
    out_bot = model(torch.cat((x_wall, y_bot), dim=1))
    out_top = model(torch.cat((x_wall, y_top), dim=1))
    loss_walls = torch.mean(out_bot[:, 0:2]**2 + out_top[:, 0:2]**2)

    out_in = model(torch.cat((x_in, y_in), dim=1))
    loss_inlet = torch.mean((out_in[:, 0:1] - u_in_exact)**2 + out_in[:, 1:2]**2)

    out_out = model(torch.cat((x_out, y_out), dim=1))
    loss_outlet = torch.mean(out_out[:, 2:3]**2)

    total_loss = physics_loss + 10.0 * (loss_walls + loss_inlet + loss_outlet)
    total_loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.6f}")

# 5. Evaluate Velocity & Pressure Fields Across the Pipe
nx, ny = 80, 40
x_lin = np.linspace(0, L_x, nx)
y_lin = np.linspace(0, L_y, ny)
X, Y = np.meshgrid(x_lin, y_lin)

test_pts = torch.tensor(np.column_stack((X.ravel(), Y.ravel())), dtype=torch.float32)
with torch.no_grad():
    preds = model(test_pts).numpy()

U_grid = preds[:, 0].reshape(ny, nx)
V_grid = preds[:, 1].reshape(ny, nx)
P_grid = preds[:, 2].reshape(ny, nx)

# Compare horizontal velocity against exact analytical Poiseuille solution
U_exact = 4.0 * Y * (1.0 - Y)
mse_u = np.mean((U_grid - U_exact) ** 2)
print(f"\nNavier-Stokes Horizontal Velocity MSE: {mse_u:.8f}")

# 6. Plot Velocity Streamlines & Pressure Drop
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6.5))

im0 = ax0.contourf(X, Y, U_grid, levels=30, cmap="plasma")
ax0.streamplot(X, Y, U_grid, V_grid, color="white", density=1.0, linewidth=0.8)
ax0.set_title("Stage 3B PINN: Navier-Stokes Pipe Velocity & Streamlines")
ax0.set_ylabel("Pipe Height y (m)")
plt.colorbar(im0, ax=ax0, label="Speed u (m/s)")

im1 = ax1.contourf(X, Y, P_grid, levels=30, cmap="coolwarm")
ax1.set_title("Discovered Fluid Pressure Gradient (High Pressure Pushes Fluid Right!)")
ax1.set_xlabel("Pipe Length x (m)")
ax1.set_ylabel("Pipe Height y (m)")
plt.colorbar(im1, ax=ax1, label="Pressure p (Pa)")

plt.tight_layout()
plt.savefig("images/stage4_navier_stokes.png", dpi=300, bbox_inches="tight")
plt.show()
