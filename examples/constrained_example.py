import numpy as np
import matplotlib.pyplot as plt
from LcGP.mogp.core import MOGPR
from LcGP.mogp.kernels.LCMKernel import LMCKernel
from LcGP.mogp.kernels.Kernel import RBFKernel

def main():
    print("Generating constrained synthetic data (y2 = -y1)...")
    
    # 1. Generate synthetic data: y1 = sin(x), y2 = -sin(x)
    np.random.seed(1337)
    n_per_output = 40
    X_train = np.sort(np.random.uniform(0, 10, size=(n_per_output, 1)), axis=0)
    
    # True functions
    y1_true = np.sin(X_train).flatten()
    y2_true = -np.sin(X_train).flatten() # Exactly opposite
    
    # Add some noise
    noise_lvl = 0.0
    y1_obs = y1_true + np.random.normal(0, noise_lvl, size=n_per_output)
    y2_obs = y2_true + np.random.normal(0, noise_lvl, size=n_per_output)
    
    # Stack for MOGPR
    Y_train = np.column_stack([y1_obs, y2_obs])
    
    # Test points
    X_test = np.linspace(0, 10, 200).reshape(-1, 1)
    y1_test_true = np.sin(X_test).flatten()
    y2_test_true = -np.sin(X_test).flatten()
    
    # 2. Setup Model
    print("Setting up MOGP model with LMC Kernel...")
    # Base spatial kernel
    rbf = RBFKernel(input_dim=1)
    
    # LMC Kernel
    # rank=1 implies that the two outputs are linear combinations of 1 latent function.
    # This is perfect for capturing y2 = -y1 (which is y2 = -1 * y1).
    kernel = LMCKernel(base_kernels=[rbf], output_dim=2, rank=[1])
    
    model = MOGPR(kernel=kernel, use_efficient_lik=True)
    
    # 3. Fit
    print("Fitting model...")
    # Using efficient likelihood since rank=1 and 1 base kernel
    model.fit(X_train, Y_train, n_restarts=5, verbose=False, use_init_pca=True)
    
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
        print("SUCCESS: Strong negative correlation captured!")
    else:
        print("WARNING: Correlation is weak or positive.")
        
    # 5. Predict
    print("\nPredicting on test grid...")
    y_pred, y_var = model.predict(X_test)
    
    y1_pred = y_pred[:, 0]
    y2_pred = y_pred[:, 1]
    y1_std = np.sqrt(y_var[:, 0])
    y2_std = np.sqrt(y_var[:, 1])
    
    # 6. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Plot y1
    axes[0].plot(X_test, y1_test_true, 'k--', label='True sin(x)')
    axes[0].scatter(X_train, y1_obs, c='r', marker='x', label='Observed y1')
    axes[0].plot(X_test, y1_pred, 'b-', label='Predicted y1')
    axes[0].fill_between(X_test.flatten(), y1_pred - 2*y1_std, y1_pred + 2*y1_std, color='b', alpha=0.2)
    axes[0].set_title("Output 1 (sin)")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot y2
    axes[1].plot(X_test, y2_test_true, 'k--', label='True -sin(x)')
    axes[1].scatter(X_train, y2_obs, c='r', marker='x', label='Observed y2')
    axes[1].plot(X_test, y2_pred, 'g-', label='Predicted y2')
    axes[1].fill_between(X_test.flatten(), y2_pred - 2*y2_std, y2_pred + 2*y2_std, color='g', alpha=0.2)
    axes[1].set_title("Output 2 (-sin)")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.suptitle(f"Constrained MOGP: Learned Correlation = {corr:.3f}")
    plt.tight_layout()
    plt.savefig('constrained_example.png')
    print("Plot saved to constrained_example.png")
    plt.show()

if __name__ == "__main__":
    main()
