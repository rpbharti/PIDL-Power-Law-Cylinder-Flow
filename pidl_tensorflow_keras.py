"""
Physics-Informed Deep Learning (PIDL) for 2D Power-Law Fluid Flow over a Cylinder
TensorFlow/Keras Implementation

Author: Physics-Informed ML
Description: This module implements a PINN using TensorFlow/Keras for steady 2D 
incompressible flow of a power-law fluid around a circular cylinder.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time

# ============================================================
# Configuration
# ============================================================
class Config:
    """Configuration parameters for PIDL simulation"""
    # Geometry
    R = 0.5                    # cylinder radius
    U_inf = 1.0                # far-field inflow speed
    
    # Fluid properties
    rho = 1000.0               # density (kg/m^3)
    m = 0.1                    # consistency index (Pa·s^n)
    n = 0.7                    # power-law index (n<1 => shear-thinning)
    eps = 1e-8                 # regularization to avoid singularity
    
    # Domain bounds
    x_min, x_max = -10 * R, 15 * R
    y_min, y_max = -8 * R, 8 * R
    
    # Training parameters
    n_iter = 8000
    n_interior = 2048
    n_bc = 512
    batch_size = 256
    learning_rate = 1e-3
    lambda_pde = 1.0           # PDE weight
    lambda_bc = 100.0          # BC weight
    lambda_interior = 0.1      # Interior continuity weight


# ============================================================
# Custom Loss Functions
# ============================================================
def compute_pde_residuals(model, xy):
    """
    Compute PDE residuals (continuity and momentum equations)
    for power-law fluid flow
    
    Returns:
        tuple: (continuity_residual, mom_x_residual, mom_y_residual)
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xy)
        u, v, p = model(xy)
        
        # First derivatives
        du_dx = tape.gradient(u, xy, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 0:1]
        du_dy = tape.gradient(u, xy, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 1:2]
        
        dv_dx = tape.gradient(v, xy, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 0:1]
        dv_dy = tape.gradient(v, xy, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 1:2]
        
        dp_dx = tape.gradient(p, xy, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 0:1]
        dp_dy = tape.gradient(p, xy, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 1:2]
    
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(xy)
        u, v, p = model(xy)
        du_dx = tape2.gradient(u, xy, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 0:1]
        du_dy = tape2.gradient(u, xy, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 1:2]
        dv_dx = tape2.gradient(v, xy, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 0:1]
        dv_dy = tape2.gradient(v, xy, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 1:2]
        
        # Strain-rate tensor
        e11 = du_dx
        e22 = dv_dy
        e12 = 0.5 * (du_dy + dv_dx)
        
        # Shear-rate magnitude
        gamma_dot = tf.sqrt(2.0 * (e11**2 + e22**2 + 2.0 * e12**2) + Config.eps)
        
        # Power-law viscosity
        eta_eff = Config.m * tf.pow(gamma_dot, Config.n - 1.0)
        
        # Stress tensor
        tau_xx = 2.0 * eta_eff * e11
        tau_xy = 2.0 * eta_eff * e12
        tau_yy = 2.0 * eta_eff * e22
        
        # Divergence of stress
        d2tau_xx_dx2 = tape2.gradient(tau_xx, xy, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 0:1]
        d2tau_xy_dy2 = tape2.gradient(tau_xy, xy, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 1:2]
        d2tau_yy_dy2 = tape2.gradient(tau_yy, xy, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 1:2]
        d2tau_xy_dx2 = tape2.gradient(tau_xy, xy, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, 0:1]
    
    # Continuity residual
    continuity = du_dx + dv_dy
    
    # Momentum residuals (simplified)
    mom_x = Config.rho * (u * du_dx + v * du_dy) + dp_dx - (d2tau_xx_dx2 + d2tau_xy_dy2)
    mom_y = Config.rho * (u * dv_dx + v * dv_dy) + dp_dy - (d2tau_xy_dx2 + d2tau_yy_dy2)
    
    return continuity, mom_x, mom_y


# ============================================================
# Custom Layers and Model
# ============================================================
class PINN_Model(keras.Model):
    """PINN model for power-law cylinder flow"""
    
    def __init__(self):
        super(PINN_Model, self).__init__()
        
        # Shared hidden layers
        self.dense1 = keras.layers.Dense(64, activation='tanh', 
                                         kernel_initializer='glorot_normal')
        self.dense2 = keras.layers.Dense(64, activation='tanh',
                                         kernel_initializer='glorot_normal')
        self.dense3 = keras.layers.Dense(64, activation='tanh',
                                         kernel_initializer='glorot_normal')
        self.dense4 = keras.layers.Dense(64, activation='tanh',
                                         kernel_initializer='glorot_normal')
        
        # Output layers for u, v, p
        self.u_out = keras.layers.Dense(1, kernel_initializer='glorot_normal')
        self.v_out = keras.layers.Dense(1, kernel_initializer='glorot_normal')
        self.p_out = keras.layers.Dense(1, kernel_initializer='glorot_normal')
    
    def call(self, xy):
        """
        Forward pass
        Args:
            xy: coordinates [x, y]
        Returns:
            u, v, p: velocity components and pressure
        """
        x = self.dense1(xy)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        
        u = self.u_out(x)
        v = self.v_out(x)
        p = self.p_out(x)
        
        return u, v, p


# ============================================================
# Data Generation
# ============================================================
def sample_interior_points(n_samples):
    """Sample points inside the domain (excluding cylinder)"""
    xs_all = []
    ys_all = []
    
    while len(xs_all) < n_samples:
        x = np.random.uniform(Config.x_min, Config.x_max, 4096)
        y = np.random.uniform(Config.y_min, Config.y_max, 4096)
        mask = (x**2 + y**2) > (1.05 * Config.R)**2
        xs_all.extend(x[mask].tolist())
        ys_all.extend(y[mask].tolist())
    
    xs = np.asarray(xs_all[:n_samples], dtype=np.float32)
    ys = np.asarray(ys_all[:n_samples], dtype=np.float32)
    
    return np.column_stack([xs, ys])


def sample_inlet(n_samples):
    """Sample points at inlet boundary"""
    y = np.random.uniform(Config.y_min, Config.y_max, n_samples)
    x = np.full(n_samples, Config.x_min)
    return np.column_stack([x, y]).astype(np.float32)


def sample_outlet(n_samples):
    """Sample points at outlet boundary"""
    y = np.random.uniform(Config.y_min, Config.y_max, n_samples)
    x = np.full(n_samples, Config.x_max)
    return np.column_stack([x, y]).astype(np.float32)


def sample_top_bottom(n_samples):
    """Sample points at top and bottom boundaries"""
    x_top = np.random.uniform(Config.x_min, Config.x_max, n_samples)
    y_top = np.full(n_samples, Config.y_max)
    
    x_bot = np.random.uniform(Config.x_min, Config.x_max, n_samples)
    y_bot = np.full(n_samples, Config.y_min)
    
    return (np.column_stack([x_top, y_top]).astype(np.float32),
            np.column_stack([x_bot, y_bot]).astype(np.float32))


def sample_cylinder(n_samples):
    """Sample points on cylinder surface (no-slip boundary)"""
    theta = np.random.uniform(0.0, 2.0 * np.pi, n_samples)
    x = Config.R * np.cos(theta)
    y = Config.R * np.sin(theta)
    return np.column_stack([x, y]).astype(np.float32)


# ============================================================
# Training Loop
# ============================================================
@tf.function
def train_step(model, optimizer, xy_interior, xy_inlet, xy_outlet, 
               xy_top, xy_bottom, xy_cylinder):
    """Single training step"""
    with tf.GradientTape() as tape:
        # PDE residuals on interior
        continuity, mom_x, mom_y = compute_pde_residuals(model, xy_interior)
        loss_pde = (tf.reduce_mean(continuity**2) + 
                   tf.reduce_mean(mom_x**2) + 
                   tf.reduce_mean(mom_y**2))
        
        # Boundary conditions
        # Inlet: u = U_inf, v = 0
        u_in, v_in, _ = model(xy_inlet)
        loss_inlet = (tf.reduce_mean((u_in - Config.U_inf)**2) + 
                     tf.reduce_mean(v_in**2))
        
        # Outlet: u ≈ U_inf, v ≈ 0
        u_out, v_out, _ = model(xy_outlet)
        loss_outlet = (tf.reduce_mean((u_out - Config.U_inf)**2) + 
                      tf.reduce_mean(v_out**2))
        
        # Top and bottom: u = U_inf, v = 0
        u_top, v_top, _ = model(xy_top)
        u_bot, v_bot, _ = model(xy_bottom)
        loss_top = (tf.reduce_mean((u_top - Config.U_inf)**2) + 
                   tf.reduce_mean(v_top**2))
        loss_bottom = (tf.reduce_mean((u_bot - Config.U_inf)**2) + 
                      tf.reduce_mean(v_bot**2))
        
        # Cylinder: no-slip (u = 0, v = 0)
        u_cyl, v_cyl, _ = model(xy_cylinder)
        loss_cylinder = (tf.reduce_mean(u_cyl**2) + 
                        tf.reduce_mean(v_cyl**2))
        
        loss_bc = (loss_inlet + loss_outlet + loss_top + 
                  loss_bottom + loss_cylinder)
        
        # Total loss
        loss_total = Config.lambda_pde * loss_pde + Config.lambda_bc * loss_bc
    
    # Backpropagation
    gradients = tape.gradient(loss_total, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss_pde, loss_bc, loss_total


def train_model(model, n_epochs=None):
    """Train the PINN model"""
    if n_epochs is None:
        n_epochs = Config.n_iter
    
    optimizer = keras.optimizers.Adam(learning_rate=Config.learning_rate)
    scheduler = keras.optimizers.schedules.ExponentialDecay(
        Config.learning_rate, 
        decay_steps=1500, 
        decay_rate=0.5
    )
    optimizer.learning_rate = scheduler
    
    losses = {'pde': [], 'bc': [], 'total': []}
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Sample points
        xy_interior = sample_interior_points(Config.n_interior)
        xy_inlet = sample_inlet(Config.n_bc)
        xy_outlet = sample_outlet(Config.n_bc)
        xy_top, xy_bottom = sample_top_bottom(Config.n_bc)
        xy_cylinder = sample_cylinder(Config.n_bc)
        
        # Convert to tensors
        xy_interior = tf.constant(xy_interior)
        xy_inlet = tf.constant(xy_inlet)
        xy_outlet = tf.constant(xy_outlet)
        xy_top = tf.constant(xy_top)
        xy_bottom = tf.constant(xy_bottom)
        xy_cylinder = tf.constant(xy_cylinder)
        
        # Training step
        loss_pde, loss_bc, loss_total = train_step(
            model, optimizer, xy_interior, xy_inlet, xy_outlet,
            xy_top, xy_bottom, xy_cylinder
        )
        
        losses['pde'].append(float(loss_pde))
        losses['bc'].append(float(loss_bc))
        losses['total'].append(float(loss_total))
        
        if (epoch + 1) % 500 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:05d} | PDE Loss: {loss_pde:.3e} | " +
                  f"BC Loss: {loss_bc:.3e} | Total Loss: {loss_total:.3e} | " +
                  f"Time: {elapsed:.1f}s")
    
    return model, losses


# ============================================================
# Prediction and Visualization
# ============================================================
def predict_field(model, nx=120, ny=120):
    """Predict velocity and pressure fields"""
    xs = np.linspace(Config.x_min, Config.x_max, nx)
    ys = np.linspace(Config.y_min, Config.y_max, ny)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    
    # Predict in batches
    batch_size = 1000
    u_all = []
    v_all = []
    p_all = []
    
    for i in range(0, len(coords), batch_size):
        coords_batch = tf.constant(coords[i:i+batch_size])
        u, v, p = model(coords_batch)
        u_all.append(u.numpy())
        v_all.append(v.numpy())
        p_all.append(p.numpy())
    
    u = np.concatenate(u_all).reshape((ny, nx))
    v = np.concatenate(v_all).reshape((ny, nx))
    p = np.concatenate(p_all).reshape((ny, nx))
    
    return xs, ys, u, v, p


def compute_drag_coefficient(model, n_samples=500):
    """
    Compute drag coefficient based on pressure and shear stress
    on cylinder surface
    """
    theta = np.linspace(0, 2*np.pi, n_samples)
    x_cyl = Config.R * np.cos(theta)
    y_cyl = Config.R * np.sin(theta)
    
    # Slightly offset for derivatives
    delta = 1e-4
    xy_inner = np.column_stack([
        (Config.R - delta) * np.cos(theta),
        (Config.R - delta) * np.sin(theta)
    ]).astype(np.float32)
    
    xy_outer = np.column_stack([
        (Config.R + delta) * np.cos(theta),
        (Config.R + delta) * np.sin(theta)
    ]).astype(np.float32)
    
    xy_cyl = np.column_stack([x_cyl, y_cyl]).astype(np.float32)
    
    with tf.GradientTape() as tape:
        xy = tf.Variable(xy_cyl)
        u, v, p = model(xy)
    
    # Approximate normal stress (pressure)
    u_cyl, v_cyl, p_cyl = model(tf.constant(xy_cyl))
    p_cyl = p_cyl.numpy().flatten()
    
    # Drag coefficient = 2 * F_d / (rho * U_inf^2 * D)
    # where D = 2*R
    D = 2.0 * Config.R
    
    # Pressure drag (integrated over cylinder)
    # For a circle, normal vector = radial direction
    n_x = np.cos(theta)
    n_y = np.sin(theta)
    
    # Force from pressure
    p_force_x = -np.trapz(p_cyl * n_x, theta) / (2 * np.pi)
    p_force_y = -np.trapz(p_cyl * n_y, theta) / (2 * np.pi)
    
    # Total drag
    C_d = 2.0 * p_force_x / (Config.rho * Config.U_inf**2 * D)
    
    return C_d, p_cyl


def plot_results(model, losses=None):
    """Plot velocity, pressure, and loss curves"""
    xs, ys, u, v, p = predict_field(model, nx=150, ny=120)
    C_d, p_cyl = compute_drag_coefficient(model)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Velocity u
    ax1 = plt.subplot(3, 3, 1)
    cs = ax1.contourf(xs, ys, u.T, levels=50, cmap='jet')
    circle = plt.Circle((0, 0), Config.R, color='k', fill=False, linewidth=2)
    ax1.add_patch(circle)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Velocity u(x,y)')
    ax1.set_aspect('equal')
    plt.colorbar(cs, ax=ax1, label='u')
    
    # Velocity v
    ax2 = plt.subplot(3, 3, 2)
    cs = ax2.contourf(xs, ys, v.T, levels=50, cmap='viridis')
    circle = plt.Circle((0, 0), Config.R, color='k', fill=False, linewidth=2)
    ax2.add_patch(circle)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Velocity v(x,y)')
    ax2.set_aspect('equal')
    plt.colorbar(cs, ax=ax2, label='v')
    
    # Pressure
    ax3 = plt.subplot(3, 3, 3)
    cs = ax3.contourf(xs, ys, p.T, levels=50, cmap='coolwarm')
    circle = plt.Circle((0, 0), Config.R, color='k', fill=False, linewidth=2)
    ax3.add_patch(circle)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Pressure p(x,y)')
    ax3.set_aspect('equal')
    plt.colorbar(cs, ax=ax3, label='p')
    
    # Velocity magnitude
    ax4 = plt.subplot(3, 3, 4)
    vel_mag = np.sqrt(u**2 + v**2)
    cs = ax4.contourf(xs, ys, vel_mag.T, levels=50, cmap='plasma')
    circle = plt.Circle((0, 0), Config.R, color='k', fill=False, linewidth=2)
    ax4.add_patch(circle)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Velocity Magnitude |V|')
    ax4.set_aspect('equal')
    plt.colorbar(cs, ax=ax4, label='|V|')
    
    # Streamlines
    ax5 = plt.subplot(3, 3, 5)
    strm = ax5.streamplot(xs, ys, u.T, v.T, color=vel_mag.T, cmap='viridis', density=2)
    circle = plt.Circle((0, 0), Config.R, color='k', fill=False, linewidth=2)
    ax5.add_patch(circle)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title('Streamlines')
    ax5.set_aspect('equal')
    
    # Vorticity
    ax6 = plt.subplot(3, 3, 6)
    # Estimate vorticity: omega = dv/dx - du/dy
    dv_dx = np.gradient(v, axis=0) / (xs[1] - xs[0])
    du_dy = np.gradient(u, axis=1) / (ys[1] - ys[0])
    vorticity = dv_dx - du_dy
    cs = ax6.contourf(xs, ys, vorticity.T, levels=50, cmap='RdBu_r')
    circle = plt.Circle((0, 0), Config.R, color='k', fill=False, linewidth=2)
    ax6.add_patch(circle)
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title('Vorticity (ω = ∂v/∂x - ∂u/∂y)')
    ax6.set_aspect('equal')
    plt.colorbar(cs, ax=ax6, label='ω')
    
    # Drag coefficient annotation
    ax7 = plt.subplot(3, 3, 7)
    ax7.text(0.1, 0.8, f'Flow Configuration:', fontsize=12, weight='bold',
            transform=ax7.transAxes)
    ax7.text(0.1, 0.7, f'  Power-law index (n): {Config.n}', fontsize=10,
            transform=ax7.transAxes)
    ax7.text(0.1, 0.6, f'  Consistency (m): {Config.m} Pa·s^n', fontsize=10,
            transform=ax7.transAxes)
    ax7.text(0.1, 0.5, f'  Reynolds (Re*): {Config.rho * Config.U_inf * 2 * Config.R / Config.m:.2f}', 
            fontsize=10, transform=ax7.transAxes)
    ax7.text(0.1, 0.4, f'  Drag Coefficient (C_d): {C_d:.4f}', fontsize=11, weight='bold',
            transform=ax7.transAxes, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax7.axis('off')
    
    # Loss curves
    if losses is not None:
        ax8 = plt.subplot(3, 3, 8)
        ax8.semilogy(losses['pde'], label='PDE Loss', linewidth=2)
        ax8.semilogy(losses['bc'], label='BC Loss', linewidth=2)
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Loss')
        ax8.set_title('Training Losses')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        ax9 = plt.subplot(3, 3, 9)
        ax9.semilogy(losses['total'], label='Total Loss', linewidth=2, color='red')
        ax9.set_xlabel('Epoch')
        ax9.set_ylabel('Loss')
        ax9.set_title('Total Loss Evolution')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pidl_tensorflow_results.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Physics-Informed Deep Learning for Power-Law Cylinder Flow")
    print("TensorFlow/Keras Implementation")
    print("=" * 60)
    
    # Create and train model
    model = PINN_Model()
    model, losses = train_model(model, n_epochs=6000)
    
    # Save model
    model.save('pidl_powerlaw_cylinder_model.h5')
    print("\nModel saved as 'pidl_powerlaw_cylinder_model.h5'")
    
    # Plot results
    plot_results(model, losses)
    
    # Compute final drag coefficient
    C_d, _ = compute_drag_coefficient(model)
    print(f"\nFinal Drag Coefficient: {C_d:.6f}")
