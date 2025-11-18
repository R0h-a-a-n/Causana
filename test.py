import numpy as np
import pandas as pd

def generate_abc_dynamics(
    n_steps: int = 300,
    noise_std: float = 0.1,
    seed: int = 42
):
    """
    Generate a time series dataset with variables a, b, c, d where:
      - increase in a causes:
            b to increase 2x (linear)
            c to increase 3x
            d to decrease 1x
    """

    np.random.seed(seed)

    # Initialize
    a = np.zeros(n_steps)
    b = np.zeros(n_steps)
    c = np.zeros(n_steps)
    d = np.zeros(n_steps)

    # Random initial values
    a[0] = np.random.randn()
    b[0] = np.random.randn()
    c[0] = np.random.randn()
    d[0] = np.random.randn()

    for t in range(1, n_steps):
        # smooth random walk for a
        a[t] = a[t-1] + np.random.normal(0, 0.5)

        # delta change in a
        delta_a = a[t] - a[t-1]

        # deterministic causal relationships
        b[t] = b[t-1] + 2.0 * delta_a + np.random.normal(0, noise_std)
        c[t] = c[t-1] + 3.0 * delta_a + np.random.normal(0, noise_std)
        d[t] = d[t-1] - 1.0 * delta_a + np.random.normal(0, noise_std)

    df = pd.DataFrame({
        "time": np.arange(n_steps),
        "a": a,
        "b": b,
        "c": c,
        "d": d
    })

    return df


if __name__ == "__main__":
    df = generate_abc_dynamics(n_steps=300, noise_std=0.12)
    out = "abcd_causal_timeseries.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out} with shape {df.shape}")
    print(df.head())
