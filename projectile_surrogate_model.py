import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


g = 9.81
FIG_DIR = "projectile_figures"
os.makedirs(FIG_DIR, exist_ok=True)


def derivatives(state, k):
    x, y, vx, vy = state
    v = np.sqrt(vx**2 + vy**2)

    dxdt = vx
    dydt = vy
    dvxdt = -k * v * vx
    dvydt = -g - k * v * vy

    return np.array([dxdt, dydt, dvxdt, dvydt])


def rk4_step(state, dt, k):
    k1 = derivatives(state, k)
    k2 = derivatives(state + 0.5 * dt * k1, k)
    k3 = derivatives(state + 0.5 * dt * k2, k)
    k4 = derivatives(state + dt * k3, k)

    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_projectile(v0, theta_deg, k, dt=0.01, max_steps=20000):
    theta = np.radians(theta_deg)

    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    state = np.array([0.0, 0.0, vx0, vy0])

    xs = [state[0]]
    ys = [state[1]]
    ts = [0.0]

    for _ in range(max_steps):
        prev_state = state.copy()
        prev_t = ts[-1]

        state = rk4_step(state, dt, k)
        current_t = prev_t + dt

        xs.append(state[0])
        ys.append(state[1])
        ts.append(current_t)

        if state[1] < 0:
            x1, y1 = prev_state[0], prev_state[1]
            x2, y2 = state[0], state[1]
            t1, t2 = prev_t, current_t

            if y2 != y1:
                alpha = -y1 / (y2 - y1)
                x_ground = x1 + alpha * (x2 - x1)
                t_ground = t1 + alpha * (t2 - t1)
            else:
                x_ground = x2
                t_ground = t2

            xs[-1] = x_ground
            ys[-1] = 0.0
            ts[-1] = t_ground
            break

    return {
        "x": np.array(xs),
        "y": np.array(ys),
        "t": np.array(ts),
        "range": xs[-1],
        "max_height": np.max(ys),
        "flight_time": ts[-1],
    }


def generate_dataset(n_samples=1000, random_seed=42):
    rng = np.random.default_rng(random_seed)
    rows = []

    for _ in range(n_samples):
        v0 = rng.uniform(15, 80)
        theta_deg = rng.uniform(20, 70)
        k = rng.uniform(0.001, 0.03)

        sim = simulate_projectile(v0, theta_deg, k)

        rows.append({
            "v0": v0,
            "theta_deg": theta_deg,
            "k": k,
            "range": sim["range"],
            "max_height": sim["max_height"],
            "flight_time": sim["flight_time"],
        })

    return pd.DataFrame(rows)


def print_metrics(y_true, y_pred, title, target_names):
    print(f"\n{title}")
    print("-" * len(title))

    for i, name in enumerate(target_names):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        print(f"\n{name}:")
        print(f"  MAE  = {mae:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  R²   = {r2:.4f}")


if __name__ == "__main__":
    target_names = ["range", "max_height", "flight_time"]

    sample_cases = [
        (30, 35, 0.005),
        (40, 45, 0.010),
        (50, 55, 0.020),
    ]

    plt.figure(figsize=(8, 6))
    for v0, theta_deg, k in sample_cases:
        sim = simulate_projectile(v0, theta_deg, k)
        label = f"v0={v0}, angle={theta_deg}°, k={k:.3f}"
        plt.plot(sim["x"], sim["y"], label=label)

    plt.xlabel("Horizontal distance x (m)")
    plt.ylabel("Vertical position y (m)")
    plt.title("Projectile Trajectories with Air Resistance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "sample_trajectories.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    df = generate_dataset(n_samples=1200, random_seed=42)

    print("First 5 rows of dataset:")
    print(df.head())
    print("\nDataset shape:", df.shape)

    df.to_csv("projectile_dataset.csv", index=False)
    print("\nDataset saved as 'projectile_dataset.csv'")

    X = df[["v0", "theta_deg", "k"]].values
    Y = df[["range", "max_height", "flight_time"]].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            max_iter=3000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42
        ))
    ])

    model.fit(X_train, Y_train)

    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    print_metrics(Y_train, Y_train_pred, "Training Performance", target_names)
    print_metrics(Y_test, Y_test_pred, "Test Performance", target_names)

    print("\nTrain vs Test R² Comparison")
    print("---------------------------")
    for i, name in enumerate(target_names):
        r2_train = r2_score(Y_train[:, i], Y_train_pred[:, i])
        r2_test = r2_score(Y_test[:, i], Y_test_pred[:, i])
        print(f"{name}: Train R² = {r2_train:.4f}, Test R² = {r2_test:.4f}")

    df_new = generate_dataset(n_samples=200, random_seed=999)

    X_new = df_new[["v0", "theta_deg", "k"]].values
    Y_new = df_new[["range", "max_height", "flight_time"]].values

    Y_new_pred = model.predict(X_new)

    print_metrics(Y_new, Y_new_pred, "Fresh Unseen Data Performance", target_names)

    for i, name in enumerate(target_names):
        plt.figure(figsize=(6, 6))
        plt.scatter(Y_test[:, i], Y_test_pred[:, i], alpha=0.7)
        min_val = min(Y_test[:, i].min(), Y_test_pred[:, i].min())
        max_val = max(Y_test[:, i].max(), Y_test_pred[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], "--")
        plt.xlabel(f"Actual {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"Predicted vs Actual: {name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIG_DIR, f"predicted_vs_actual_{name}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()
        plt.close()

    for i, name in enumerate(target_names):
        residuals = Y_test[:, i] - Y_test_pred[:, i]

        plt.figure(figsize=(7, 5))
        plt.scatter(Y_test[:, i], residuals, alpha=0.7)
        plt.axhline(0, linestyle="--")
        plt.xlabel(f"Actual {name}")
        plt.ylabel("Residual (Actual - Predicted)")
        plt.title(f"Residual Plot ({name})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIG_DIR, f"residual_plot_{name}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()
        plt.close()

    for i, name in enumerate(target_names):
        residuals = Y_test[:, i] - Y_test_pred[:, i]

        plt.figure(figsize=(7, 5))
        plt.hist(residuals, bins=25, edgecolor="black")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.title(f"Residual Histogram ({name})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIG_DIR, f"residual_histogram_{name}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()
        plt.close()

    new_input = np.array([[45.0, 50.0, 0.0120]])
    pred = model.predict(new_input)[0]

    print("\nExample Prediction for One New Input")
    print("------------------------------------")
    print("Input:")
    print(f"  v0 = {new_input[0, 0]:.2f} m/s")
    print(f"  angle = {new_input[0, 1]:.2f} degrees")
    print(f"  k = {new_input[0, 2]:.4f}")

    print("Predicted outputs:")
    print(f"  range       = {pred[0]:.3f} m")
    print(f"  max_height  = {pred[1]:.3f} m")
    print(f"  flight_time = {pred[2]:.3f} s")

    true_sim = simulate_projectile(v0=45.0, theta_deg=50.0, k=0.0120)

    print("\nTrue RK4 simulation outputs:")
    print(f"  range       = {true_sim['range']:.3f} m")
    print(f"  max_height  = {true_sim['max_height']:.3f} m")
    print(f"  flight_time = {true_sim['flight_time']:.3f} s")