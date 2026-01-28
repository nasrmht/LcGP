import numpy as np
import matplotlib.pyplot as plt
from LcGP.mogp.core import MOGPR
from LcGP.mogp.kernels.LCMKernel import LMCKernel
from LcGP.mogp.kernels.Kernel import RBFKernel

def main():
    print("Generating constrained synthetic data (y2 = -y1)...")
    
    # 1. Generate synthetic data: y1 = sin(x), y2 = -sin(x)
    np.random.seed(1337)
    n_per_output = 20
    X_train = np.sort(np.random.uniform(0, 10, size=(n_per_output, 1)), axis=0)
    
    # True functions
    y1_true = np.sin(X_train).flatten()
    y2_true = -np.sin(X_train).flatten() # Exactly opposite
    
    # Add some noise
    noise_lvl = 0.05
    y1_obs = y1_true + np.random.normal(0, noise_lvl, size=n_per_output)
    y2_obs = y2_true + np.random.normal(0, noise_lvl, size=n_per_output)
    
    # Stack for MOGPR
    Y_train = np.column_stack([y1_obs, y2_obs])
    
    # 2. Setup Model
    print("Setting up MOGP model with LMC Kernel...")
    rbf = RBFKernel(input_dim=1)
    kernel = LMCKernel(base_kernels=[rbf], output_dim=2, rank=[1])
    
    model = MOGPR(kernel=kernel, use_efficient_lik=True)
    
    # 3. Fit with PCA Init
    print("Fitting model using PCA initialization...")
    try:
        model.fit(X_train, Y_train, n_restarts=5, verbose=False, use_init_pca=True)
    except Exception as e:
        print(f"FAILED: {e}")
        raise e
    
    # 4. Analyze Learned Coregionalization Matrix B
    print("\n--- Learned Coregionalization ---")
    B = kernel.get_B(0)
    print("Matrix B (approx):")
    print(B)
    
    # Check correlation
    cov_12 = B[0, 1]
    var_1 = B[0, 0]
    var_2 = B[1, 1]
    corr = cov_12 / np.sqrt(var_1 * var_2)
    print(f"Learned correlation between outputs: {corr:.4f}")
    
    if corr < -0.9:
        print("SUCCESS: Strong negative correlation captured with PCA init!")
    else:
        print("WARNING: Correlation is weak or positive.")

if __name__ == "__main__":
    main()
