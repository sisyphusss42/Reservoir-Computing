import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import Ridge
from mpl_toolkits.mplot3d import Axes3D

# Lorenz system
sigma, rho, beta = 10, 28, 8/3
def lorenz(t, state):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

# Solve Lorenz system
T = 150
dt = 0.02
num_steps = int(T / dt)
t_span = [0, T]
t_eval = np.linspace(0, T, num_steps)
initial_state = [1.0, 1.0, 1.0]
sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval).y.T

# Echo State Network parameters
n_reservoir = 500
sparsity = 0.05
spectral_radius = 0.9
leak_rate = 0.5
ridge_alpha = 1e-3

# Initialize reservoir weights
np.random.seed(42)
W_in = np.random.rand(n_reservoir, 3) * 2 - 1
W_res = np.random.rand(n_reservoir, n_reservoir) - 0.5
mask = np.random.rand(n_reservoir, n_reservoir) < sparsity
W_res *= mask

# Normalize spectral radius
eigenvalues = np.linalg.eigvals(W_res)
W_res /= np.max(np.abs(eigenvalues)) / spectral_radius

# Training phase
states = np.zeros((num_steps, n_reservoir))
state = np.zeros(n_reservoir)
for i in range(num_steps):
    state = (1 - leak_rate) * state + leak_rate * np.tanh(0.1 * (W_in @ sol[i] + W_res @ state))
    states[i] = state

# Train output weights using Ridge regression
train_len = num_steps // 15 * 14
reg = Ridge(alpha=ridge_alpha)
reg.fit(states[:train_len], sol[:train_len])

# Prediction phase
predictions = np.zeros((num_steps - train_len, 3))
state = states[train_len - 1]
input_signal = sol[train_len - 1]

for i in range(num_steps - train_len):
    state = (1 - leak_rate) * state + leak_rate * np.tanh(0.1 * (W_in @ input_signal + W_res @ state))
    predictions[i] = reg.predict(state.reshape(1, -1))
    input_signal = predictions[i]  # Using predicted values for next iteration

# Create figure with subplots
plt.figure(figsize=(15, 6))

# 2D plot
plt.subplot(121)
plt.plot(t_eval[train_len:], sol[train_len:, 0], label="True X")
plt.plot(t_eval[train_len:], predictions[:, 0], '--', label="Predicted X")
plt.legend(loc='upper left')
plt.title("Lorenz System X-Component Prediction")
plt.xlabel("Time")
plt.ylabel("X Value")

# Add text box with ESN parameters at the bottom right
params_text = (
    f"n_reservoir: {n_reservoir}\n"
    f"sparsity: {sparsity}\n"
    f"spectral_radius: {spectral_radius}\n"
    f"leak_rate: {leak_rate}\n"
    f"ridge_alpha: {ridge_alpha}"
)
plt.gca().text(0.98, 0.02, params_text, transform=plt.gca().transAxes, 
               fontsize=10, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(facecolor='white', alpha=0.6))

# 3D plot
ax = plt.subplot(122, projection='3d')
# Plot true trajectory
ax.plot(sol[train_len:, 0], sol[train_len:, 1], sol[train_len:, 2], 
        label='True Trajectory', linewidth=1)
# Plot predicted trajectory
ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], '--',
        label='Predicted Trajectory', linewidth=1)
ax.set_title("Lorenz Attractor: True vs Predicted")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.tight_layout()
plt.show()


