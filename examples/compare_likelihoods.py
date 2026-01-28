import numpy as np
import matplotlib.pyplot as plt
import time
from LcGP.mogp.core import MOGPR
from LcGP.mogp.kernels.LCMKernel import LMCKernel
from LcGP.mogp.kernels.Kernel import RBFKernel

def main():
    print("=== Comparing Efficient vs Naive Likelihood Implementations ===")
    
    # 1. Generate constrained synthetic data (y2 = -y1)
    np.random.seed(1337)
    n_per_output = 20
    X_train = np.sort(np.random.uniform(0, 10, size=(n_per_output, 1)), axis=0)
    
    # True functions
    y1_true = np.sin(X_train).flatten()
    y2_true = -np.sin(X_train).flatten() 
    
    # Add noise
    noise_lvl = 0.05
    y1_obs = y1_true + np.random.normal(0, noise_lvl, size=n_per_output)
    y2_obs = y2_true + np.random.normal(0, noise_lvl, size=n_per_output)
    
    Y_train = np.column_stack([y1_obs, y2_obs])
    X_test = np.linspace(0, 10, 50).reshape(-1, 1)

    # 2. Setup Models
    print("\n--- Model Setup ---")
    # We must ensure both start from relatively similar conditions or use enough restarts
    n_restarts = 10
    seed = 42
    
    # Model 1: Efficient
    rbf1 = RBFKernel(input_dim=1)
    kernel1 = LMCKernel(base_kernels=[rbf1], output_dim=2, rank=[1])
    model_eff = MOGPR(kernel=kernel1, use_efficient_lik=True, verbose=False)
    
    # Model 2: Naive
    rbf2 = RBFKernel(input_dim=1)
    kernel2 = LMCKernel(base_kernels=[rbf2], output_dim=2, rank=[1])
    model_naive = MOGPR(kernel=kernel2, use_efficient_lik=False, verbose=False) # Naive
    
    # 3. Fit
    print(f"Fitting Efficient model (n_restarts={n_restarts})...")
    start = time.time()
    model_eff.fit(X_train, Y_train, n_restarts=n_restarts, seed=seed, use_init_pca=True)
    time_eff = time.time() - start
    
    print(f"Fitting Naive model (n_restarts={n_restarts})...")
    start = time.time()
    model_naive.fit(X_train, Y_train, n_restarts=n_restarts, seed=seed, use_init_pca=True)
    time_naive = time.time() - start
    
    # 4. Compare Results
    print("\n=== Results Comparison ===")
    
    # NLL
    nll_eff = model_eff.log_marginal_likelihood()
    nll_naive = model_naive.log_marginal_likelihood()
    
    print(f"{'Metric':<20} | {'Efficient':<15} | {'Naive':<15} | {'Diff':<15}")
    print("-" * 75)
    print(f"{'Time (s)':<20} | {time_eff:<15.4f} | {time_naive:<15.4f} | {time_naive - time_eff:<15.4f}")
    print(f"{'Log Marginal Lik':<20} | {nll_eff:<15.4f} | {nll_naive:<15.4f} | {abs(nll_eff - nll_naive):<15.4e}")
    
    # Parameters
    params_eff = model_eff.kernel.params
    params_naive = model_naive.kernel.params
    # Note: params might be permuted or have different signs if latent functions are flipped, 
    # but B matrices should be similar.
    
    B_eff = model_eff.kernel.get_B(0)
    B_naive = model_naive.kernel.get_B(0)
    
    print(f"\n--- Matrix B Comparison ---")
    print("Efficient B:\n", B_eff)
    print("Naive B:\n", B_naive)
    print("Diff Norm:", np.linalg.norm(B_eff - B_naive))
    
    # Correlations
    def get_corr(B):
        return B[0, 1] / np.sqrt(B[0, 0] * B[1, 1])
        
    corr_eff = get_corr(B_eff)
    corr_naive = get_corr(B_naive)
    
    print(f"\n{'Correlation':<20} | {corr_eff:<15.4f} | {corr_naive:<15.4f}")
    
    # Noise
    noise_eff = np.exp(model_eff.log_noise_variance)
    noise_naive = np.exp(model_naive.log_noise_variance)
    print(f"{'Noise Var':<20} | {noise_eff:<15.6f} | {noise_naive:<15.6f}")

    # Predictions
    print("\n--- Predictions Comparison ---")
    y_pred_eff, _ = model_eff.predict(X_test)
    y_pred_naive, _ = model_naive.predict(X_test)
    
    pred_diff = np.linalg.norm(y_pred_eff - y_pred_naive)
    print(f"Prediction Difference (L2 norm): {pred_diff:.6f}")
    
    if abs(nll_eff - nll_naive) < 1e-1:
        print("\nCONCLUSION: Both methods converged to similar solutions.")
    else:
        print("\nCONCLUSION: Methods converged to DIFFERENT solutions.")
        
    # Plot comparison
    plt.figure(figsize=(10, 5))
    plt.plot(X_test, y_pred_eff[:, 0], 'b-', label='Efficient y1')
    plt.plot(X_test, y_pred_naive[:, 0], 'r--', label='Naive y1')
    plt.scatter(X_train, y1_obs, c='k', marker='.', label='Data')
    plt.legend()
    plt.title(f"Prediction Comparison (Diff: {pred_diff:.4f})")
    plt.savefig('comparison_plot.png')
    plt.show()
    print("Comparison plot saved to comparison_plot.png")

if __name__ == "__main__":
    main()
