import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
import time

# Set random seed for reproducibility
rpy.set_seed(42)
rpy.verbosity(0)

# Parameters for the Lorenz system
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Lorenz system differential equations
def lorenz_system(t, xyz):
    x, y, z = xyz
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Generate Lorenz system data
def generate_lorenz_data(initial_state, t_span, dt):
    t_eval = np.arange(t_span[0], t_span[1], dt)
    solution = solve_ivp(
        lorenz_system, 
        t_span, 
        initial_state, 
        t_eval=t_eval, 
        method='RK45'
    )
    return solution.y.T  # Shape: (timesteps, 3)

# Generate training and testing data
initial_state = [1.0, 1.0, 1.0]
train_span = (0, 50)
test_span = (50, 60)
dt = 0.01

# Generate data
train_data = generate_lorenz_data(initial_state, train_span, dt)
test_initial = train_data[-1]
test_data = generate_lorenz_data(test_initial, test_span, dt)

# Normalize data
data_mean = np.mean(train_data, axis=0)
data_std = np.std(train_data, axis=0)
train_data_normalized = (train_data - data_mean) / data_std
test_data_normalized = (test_data - data_mean) / data_std

# Prepare training data for ESN
# X: input data, Y: target data (one step ahead prediction)
X_train = train_data_normalized[:-1]
Y_train = train_data_normalized[1:]

# Create ESN components
reservoir = Reservoir(
    units=50,             # Number of reservoir neurons
    lr=0.5,                # Leaking rate
    sr=1.0,               # Spectral radius
    input_scaling=0.1,     # Scaling of the input weights
    bias_scaling=0.5,      # Scaling of the bias
    rc_connectivity=0.05    # Percentage of non-zero connections in the reservoir
)

readout = Ridge(ridge=1e-6)  # Regularization parameter

# Create the ESN model
esn = reservoir >> readout

# Train the ESN
# Using a warmup period to let the reservoir dynamics stabilize
warmup = 100
start_time = time.time()
esn = esn.fit(X_train, Y_train, warmup=warmup)
training_time = time.time() - start_time

# Function for autonomous prediction
def predict_autonomously(model, initial_data, n_steps):
    predictions = np.zeros((n_steps, initial_data.shape[1]))
    current_input = initial_data.reshape(1, -1)
    
    for i in range(n_steps):
        prediction = model.run(current_input)
        predictions[i] = prediction
        current_input = prediction  # Use prediction as next input
        
    return predictions

# Perform autonomous prediction on test data
# Starting with the last point of the training data
prediction_steps = len(test_data_normalized)
initial_point = train_data_normalized[-1].reshape(1, -1)
predictions_normalized = predict_autonomously(esn, initial_point, prediction_steps)

# Denormalize predictions
predictions = predictions_normalized * data_std + data_mean

# Evaluate prediction using RMSE
mse = np.mean((predictions - test_data)**2)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Calculate RMSE for each component separately
rmse_components = np.sqrt(np.mean((predictions - test_data)**2, axis=0))
print(f"RMSE for x component: {rmse_components[0]}")
print(f"RMSE for y component: {rmse_components[1]}")
print(f"RMSE for z component: {rmse_components[2]}")

# Visualize results
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
component_names = ['x', 'y', 'z']

for i, ax in enumerate(axes):
    ax.plot(test_data[:, i], 'b-', linewidth=1.5, label='True')
    ax.plot(predictions[:, i], 'r--', linewidth=1.5, label='Predicted')
    
    ax.set_title(f'Lorenz {component_names[i]} Component', fontsize=12)
    ax.set_ylabel(f'{component_names[i]}(t)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')

axes[-1].set_xlabel('Time Steps', fontsize=10)
plt.tight_layout()
plt.savefig('lorenz_components.png', dpi=300)
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for i, ax in enumerate(axes):
    ax.plot(np.abs(test_data[:, i] - predictions[:, i]), 'g-', linewidth=1.5)
    ax.set_title(f'Absolute Error in {component_names[i]} Component', fontsize=12)
    ax.set_ylabel(f'|Error in {component_names[i]}|', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

axes[-1].set_xlabel('Time Steps', fontsize=10)
plt.tight_layout()
plt.savefig('lorenz_errors.png', dpi=300)
plt.show()

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# True trajectory in blue
ax.plot(test_data[:, 0], test_data[:, 1], test_data[:, 2], 'b-', linewidth=1.5, label='True')
# Predicted trajectory in orange
ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], color='orange', linestyle='-', linewidth=1.5, label='Predicted')

# Mark starting points
ax.scatter(test_data[0, 0], test_data[0, 1], test_data[0, 2], color='blue', s=50, label='True Start')
ax.scatter(predictions[0, 0], predictions[0, 1], predictions[0, 2], color='orange', s=50, label='Prediction Start')

ax.set_title('Lorenz Attractor: True vs Predicted', fontsize=14)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)
ax.legend(loc='upper right')

ax.view_init(elev=20, azim=120)

reservoir_params = {
    "reservoir_size": reservoir.hypers["units"],
    "sparsity": reservoir.hypers["rc_connectivity"],
    "spectral_radius": reservoir.hypers["sr"],
    "leaking rate": reservoir.hypers["lr"],
    "input_scaling": reservoir.hypers["input_scaling"],
    "bias_scaling": reservoir.hypers["bias_scaling"]
}

ridge_param = readout.hypers["ridge"]

model_params_text = ""
for i, (param_name, param_value) in enumerate(reservoir_params.items(), 1):
    model_params_text += f"{i}. {param_name}: {param_value}\n"

model_params_text += f"{len(reservoir_params) + 1}. ridge: {ridge_param}"

# performance metrics box
performance_text = f"Training Time: {training_time:.3f}s\nRoot Mean Squared Error: {rmse:.6f}"

# parameter text box
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
fig.text(0.15, 0.15, model_params_text, fontsize=11, 
         bbox=props, transform=fig.transFigure, verticalalignment='bottom')

perf_props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
fig.text(0.15, 0.08, performance_text, fontsize=11, 
         bbox=perf_props, transform=fig.transFigure, verticalalignment='bottom')

plt.tight_layout()
plt.savefig('lorenz_3d_comparison.png', dpi=300)
plt.show()

# prediction quality over time
plt.figure(figsize=(12, 6))

# Cumulative error over time
plt.subplot(1, 2, 1)
cumulative_error = np.sqrt(np.sum((test_data - predictions)**2, axis=1))  # Euclidean distance at each time step
plt.plot(cumulative_error, 'r-', linewidth=2)
plt.title('Prediction Error Over Time', fontsize=12)
plt.xlabel('Time Steps', fontsize=10)
plt.ylabel('Euclidean Distance', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Distribution of errors
plt.subplot(1, 2, 2)
plt.hist(cumulative_error, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Prediction Errors', fontsize=12)
plt.xlabel('Error Magnitude', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('lorenz_error_analysis.png', dpi=300)
plt.show()
