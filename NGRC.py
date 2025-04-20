import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations_with_replacement

##
## Parameters
##

# time step
dt = 0.01
# units of time to warm up NVAR
warmup = 5.0
# units of time to train for
traintime = 45.0
# units of time to test for
testtime = 10.0
# total time to run for
maxtime = warmup + traintime + testtime

# calculate
warmup_pts = round(warmup/dt)
traintime_pts = round(traintime/dt)
warmtrain_pts = warmup_pts + traintime_pts
testtime_pts = round(testtime/dt)
maxtime_pts = round(maxtime/dt)

# input dimension
d = 3
# number of time delay taps
k = 3
# size of linear part of feature vector
dlin = k*d
# polynomial order for nonlinear features
p = 3

# ridge parameter for regression
ridge_param = 1e-5

# enable normalization
enable_normalization = True

# t values for whole evaluation time
t_eval = np.linspace(0, maxtime, maxtime_pts+1)

##
## Function to generate nonlinear features of arbitrary polynomial order
##

def generate_feature_vector(x_linear, p_order, normalization_factors=None):
    """
    Generate feature vector with polynomial terms up to specified order
    with optional normalization for numerical stability
    
    Parameters:
    x_linear - Linear feature vector (shape: dlin)
    p_order - Maximum polynomial order (1 = linear, 2 = quadratic, etc.)
    normalization_factors - Factors to normalize each term (if None, no normalization)
    
    Returns:
    Full feature vector with constant term, linear terms, and nonlinear terms up to p_order
    """
    # Start with constant term
    features = [1.0]
    
    # Add linear terms - apply normalization if factors are provided
    if normalization_factors is not None and enable_normalization:
        # Apply normalization for linear terms
        linear_terms = x_linear / normalization_factors[:len(x_linear)]
        features.extend(linear_terms)
    else:
        features.extend(x_linear)
    
    # Add higher-order terms
    feature_idx = len(x_linear) + 1  # Start after constant and linear terms
    
    for order in range(2, p_order + 1):
        # Generate all combinations with replacement for current order
        for indices in combinations_with_replacement(range(len(x_linear)), order):
            # Compute the product term
            if normalization_factors is not None and enable_normalization:
                # Normalized version to prevent overflow
                term = 1.0
                for idx in indices:
                    # Apply normalization factor for each variable in the product
                    term *= (x_linear[idx] / normalization_factors[idx])
                    
                    # Check for overflow and underflow
                    if not np.isfinite(term):
                        term = 0.0  # Reset to zero if we get overflow
                        break
            else:
                # Unnormalized version
                term = 1.0
                for idx in indices:
                    term *= x_linear[idx]
                    
                    # Check for overflow
                    if not np.isfinite(term):
                        term = 0.0  # Reset to zero if we get overflow
                        break
                        
            features.append(term)
            feature_idx += 1
    
    return np.array(features)

##
## Lorenz
##

sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

def lorenz_system(t, y):
    dy0 = sigma * (y[1] - y[0])
    dy1 = y[0] * (rho - y[2]) - y[1]
    dy2 = y[0] * y[1] - beta * y[2]
    
    return [dy0, dy1, dy2]


initial_state = [-6.68833468,-8.55042147,21.72678511]
lorenz_soln = solve_ivp(lorenz_system, (0, maxtime), initial_state, t_eval=t_eval, method='RK23')

# Get data
full_data = lorenz_soln.y.T  # Shape: (timesteps, 3)
train_data = full_data[warmup_pts:warmtrain_pts]

##
## NGRC
##

# Start timer for training
start_time = time.time()

# Create an array to hold the linear part of the feature vector
x = np.zeros((dlin, maxtime_pts+1))  # Added +1 to match t_eval dimension

# Fill in the linear part of the feature vector for all times
for delay in range(k):
    for j in range(delay, maxtime_pts+1):  # Adjusted range to match t_eval
        x[d*delay:d*(delay+1), j] = lorenz_soln.y[:, j-delay]

# Calculate number of features based on polynomial order p
dtot = 1  # Constant term
for order in range(1, p+1):
    dtot += len(list(combinations_with_replacement(range(dlin), order)))

print(f"Using polynomial order p={p} with {dtot} total features")

# Calculate normalization factors for numerical stability
if enable_normalization:
    # Compute the absolute max value for each variable in the training set
    # Add small epsilon to avoid division by zero
    normalization_factors = np.max(np.abs(x[:, warmup_pts:warmtrain_pts]), axis=1) + 1e-6
    print("Normalization factors:", normalization_factors)
else:
    normalization_factors = None

# Create an array to hold the full feature vector for training time
out_train = np.zeros((dtot, traintime_pts))

# Fill in the feature vector for all training times
for i in range(traintime_pts):
    out_train[:, i] = generate_feature_vector(x[:, warmup_pts+i], p, normalization_factors)

# Check if we have any numerical issues in the feature vector
if not np.all(np.isfinite(out_train)):
    print("Warning: Non-finite values in training feature vector, replacing with zeros")
    out_train[~np.isfinite(out_train)] = 0.0

# Ridge regression: train W_out to map out_train to Lorenz[t+1] - Lorenz[t]
delta_x = x[0:d, warmup_pts+1:warmtrain_pts+1] - x[0:d, warmup_pts:warmtrain_pts]
W_out = delta_x @ out_train[:, :].T @ np.linalg.pinv(out_train[:, :] @ out_train[:, :].T + ridge_param*np.identity(dtot))

# End timer for training
training_time = time.time() - start_time
print(f"Training completed in {training_time:.4f} seconds")

# Apply W_out to the training feature vector to get the training output
x_predict = x[0:d, warmup_pts:warmtrain_pts] + W_out @ out_train[:, 0:traintime_pts]

# Create a place to store feature vectors for prediction
x_test = np.zeros((dlin, testtime_pts+1))  # Add one more time point for initial condition

# Copy over initial linear feature vector (last state from training)
x_test[:, 0] = x[:, warmtrain_pts]

# Do prediction with better numerical stability
for j in range(testtime_pts):
    # Generate feature vector for current state with normalization
    out_test = generate_feature_vector(x_test[:, j], p, normalization_factors)
    
    # Check for numerical issues
    if not np.all(np.isfinite(out_test)):
        print(f"Warning: Non-finite feature vector at step {j}, replacing NaN values")
        out_test[~np.isfinite(out_test)] = 0.0
    
    # Update next state
    if j < testtime_pts:
        # Fill in the delay taps of the next state
        if j+1 < testtime_pts:
            x_test[d:dlin, j+1] = x_test[0:(dlin-d), j]
        
        # Make prediction for next state and check for numerical issues
        delta = W_out @ out_test
        if np.all(np.isfinite(delta)):
            x_test[0:d, j+1] = x_test[0:d, j] + delta
        else:
            print(f"Warning: Non-finite prediction at step {j}, using previous value")
            if j > 0:
                x_test[0:d, j+1] = x_test[0:d, j]
            else:
                # Just use a reasonable value from the Lorenz attractor if prediction fails at the first step
                x_test[0:d, j+1] = np.array([0.0, 1.0, 20.0])

# Get test data and predictions in the same format as RC.py
test_actual = full_data[warmtrain_pts+1:warmtrain_pts+testtime_pts+1]
test_predictions = x_test[0:d, 1:testtime_pts+1].T

# Calculate RMSE for test data (overall and by component)
# Avoid NaN by checking for valid predictions
valid_indices = np.all(np.isfinite(test_predictions), axis=1)
if np.any(valid_indices):
    test_rmse = np.sqrt(np.mean((test_actual[valid_indices] - test_predictions[valid_indices])**2))
    rmse_components = np.sqrt(np.mean((test_actual[valid_indices] - test_predictions[valid_indices])**2, axis=0))
    print(f"Test Root Mean Squared Error: {test_rmse}")
    print(f"RMSE for x component: {rmse_components[0]}")
    print(f"RMSE for y component: {rmse_components[1]}")
    print(f"RMSE for z component: {rmse_components[2]}")
    print(f"Prediction accuracy: {len(valid_indices)}/{len(test_predictions)} valid predictions ({100*np.sum(valid_indices)/len(valid_indices):.1f}%)")
else:
    print("Warning: No valid predictions for RMSE calculation")

##
## Plot
##

# Plot individual components (x, y, z) with cleaner visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
component_names = ['x', 'y', 'z']

for i, ax in enumerate(axes):
    # Plot actual vs predicted values for each component
    ax.plot(test_actual[:, i], 'b-', linewidth=1.5, label='True')
    ax.plot(test_predictions[:, i], 'r--', linewidth=1.5, label='Predicted')
    
    # Better styling for cleaner appearance
    ax.set_title(f'Lorenz {component_names[i]} Component (NGRC p={p})', fontsize=12)
    ax.set_ylabel(f'{component_names[i]}(t)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')

axes[-1].set_xlabel('Time Steps', fontsize=10)
plt.tight_layout()
plt.savefig(f'ngrc_lorenz_components_p{p}.png', dpi=300)
plt.show()

# Plot error separately for clarity
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for i, ax in enumerate(axes):
    # Calculate error safely avoiding NaN issues
    error = np.abs(test_actual[:, i] - test_predictions[:, i])
    error[~np.isfinite(error)] = np.nan  # Replace non-finite values with NaN for plotting
    
    # Plot absolute error
    ax.plot(error, 'g-', linewidth=1.5)
    ax.set_title(f'Absolute Error in {component_names[i]} Component (NGRC p={p})', fontsize=12)
    ax.set_ylabel(f'|Error in {component_names[i]}|', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

axes[-1].set_xlabel('Time Steps', fontsize=10)
plt.tight_layout()
plt.savefig(f'ngrc_lorenz_errors_p{p}.png', dpi=300)
plt.show()

# 3D plot with both true and predicted trajectories
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Filter out any NaN values for plotting
valid_mask = np.all(np.isfinite(test_predictions), axis=1)
valid_actual = test_actual[valid_mask]
valid_predictions = test_predictions[valid_mask]

if len(valid_predictions) > 0:
    # True trajectory in blue
    ax.plot(valid_actual[:, 0], valid_actual[:, 1], valid_actual[:, 2], 'b-', linewidth=1.5, label='True')
    # Predicted trajectory in orange
    ax.plot(valid_predictions[:, 0], valid_predictions[:, 1], valid_predictions[:, 2], color='orange', linestyle='-', linewidth=1.5, label='Predicted')

    # Mark starting points
    ax.scatter(valid_actual[0, 0], valid_actual[0, 1], valid_actual[0, 2], color='blue', s=50, label='True Start')
    ax.scatter(valid_predictions[0, 0], valid_predictions[0, 1], valid_predictions[0, 2], color='orange', s=50, label='Prediction Start')

ax.set_title(f'Lorenz Attractor: True vs Predicted (NGRC p={p})', fontsize=14)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)
ax.legend(loc='upper right')

# Set better viewing angle to show the classic butterfly shape
ax.view_init(elev=20, azim=120)

# Build parameters text for NGRC
ngrc_params = {
    "polynomial_order (p)": p,
    "time_delays (k)": k,
    "ridge_param": ridge_param,
    "feature vector size": dtot
}

# Build model parameters text dynamically
model_params_text = "\n".join([f"{i}. {param_name}: {param_value}" 
                            for i, (param_name, param_value) in enumerate(ngrc_params.items(), 1)])

# Create a separate performance metrics box
if 'test_rmse' in locals() and np.isfinite(test_rmse):
    performance_text = f"Training Time: {training_time:.3f}s\nRoot Mean Squared Error: {test_rmse:.6f}"
else:
    performance_text = f"Training Time: {training_time:.3f}s\nRoot Mean Squared Error: N/A (numerical issues)"

# Position the parameter text box
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
fig.text(0.15, 0.15, model_params_text, fontsize=11, 
         bbox=props, transform=fig.transFigure, verticalalignment='bottom')

# Position the performance metrics box
perf_props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
fig.text(0.15, 0.08, performance_text, fontsize=11, 
         bbox=perf_props, transform=fig.transFigure, verticalalignment='bottom')

plt.tight_layout()
plt.savefig(f'ngrc_lorenz_3d_comparison_p{p}.png', dpi=300)
plt.show()
