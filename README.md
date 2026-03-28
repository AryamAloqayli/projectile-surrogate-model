# projectile-surrogate-model
Physics-based simulation of projectile motion with air resistance using RK4, combined with a neural network surrogate model for fast prediction of range, height, and flight time.
# Projectile Motion with Drag + Neural Network Surrogate

This project combines numerical simulation and machine learning to model projectile motion with air resistance.  
A physics-based RK4 solver is used to generate data, and a neural network is trained to approximate the system behavior.

---

## Overview

Inputs:
- Initial speed (v0)
- Launch angle (θ)
- Drag coefficient (k)

Outputs:
- Range
- Maximum height
- Flight time

The goal is to replace repeated numerical simulations with a fast surrogate model.

---

## Physics Model

The motion is governed by:

- dx/dt = vx  
- dy/dt = vy  
- dvx/dt = -k v vx  
- dvy/dt = -g - k v vy  

where v = √(vx² + vy²)

The system is solved using the Runge-Kutta 4th order method (RK4).

---

## Dataset

- 1200 simulations generated
- Randomized inputs:
  - v0 ∈ [15, 80] m/s  
  - θ ∈ [20°, 70°]  
  - k ∈ [0.001, 0.03]

Each sample includes:
- range
- max height
- flight time

---

## Model

A neural network (MLPRegressor) is used:

- Input: (v0, θ, k)
- Output: (range, height, time)
- Architecture: 2 hidden layers (64, 64)
- StandardScaler applied to inputs

---

## Performance

### Test Performance

- Range: R² ≈ 0.9968  
- Max height: R² ≈ 0.9916  
- Flight time: R² ≈ 0.9914  

The model generalizes well, with very small differences between training and test scores.

### Fresh Unseen Data

Performance remains consistent on new data:
- R² stays above 0.99 for all outputs

This indicates strong generalization and no clear overfitting.

---

## Results & Analysis

### Projectile Trajectories

The simulated trajectories show the effect of drag:
- Higher drag reduces range and height
- Trajectories are no longer symmetric
- Motion deviates from ideal parabolic behavior

---

### Predicted vs Actual

All three outputs show strong agreement with the true simulation:

- Points lie close to the diagonal → high accuracy  
- Slight deviations appear at extreme values  

Observation:
- The model performs best in the mid-range where most data exists  
- At very large values, predictions slightly deviate  

---

### Residual Plots

Residuals are centered around zero:

- No strong systematic pattern → model captures main physics  
- Spread increases at higher values  

This suggests:
- Error grows in more extreme regimes  
- Likely due to increased nonlinearity from drag  

---

### Residual Histograms

- Errors are mostly centered near zero → low bias  
- Distribution is roughly symmetric  
- Some outliers exist, especially for large values  

Observation:
- Range and max height show heavier tails  
- Indicates occasional larger errors in high-energy cases  

---

## Example Prediction

Input:
v0=45 m/s
theta=50 degrees
k=0.012


Prediction vs simulation:

| Quantity     | Predicted | RK4 (True) |
|--------------|----------|------------|
| Range        | 77.556 m | 76.958 m   |
| Max Height   | 31.821 m | 32.585 m   |
| Flight Time  | 5.175 s  | 5.117 s    |

The prediction is very close to the true numerical solution.

---

## Key Insight

The neural network successfully learns a nonlinear mapping between inputs and outputs, replacing the need to solve differential equations each time.

This approach is useful when:
- simulations are expensive  
- fast predictions are required  
- exploring large parameter spaces  

