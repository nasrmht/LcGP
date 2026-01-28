import numpy as np
import matplotlib.pyplot as plt
from LcGP.mogp.core import MOGPR
from LcGP.mogp.kernels.ConstrainedLMCKernel import LMCKernelConstrained
from LcGP.mogp.kernels.Kernel import RBFKernel

def main():
    print("=== Testing LMCKernelConstrained ===")
    
    # 1. Generate constrained synthetic data (y1 + y2 + y3 = 0)
    np.random.seed(42)
    n_per_output = 20
    X_train = np.sort(np.random.uniform(0, 10, size=(n_per_output, 1)), axis=0)
    
    # functions
    y1_true = np.sin(X_train).flatten()
    y2_true = np.cos(X_train).flatten()
    y3_true = -y1_true - y2_true 
    
    # Add noise
    noise_lvl = 0.00 # Low noise to verify constraint easily
    y1_obs = y1_true + np.random.normal(0, noise_lvl, size=n_per_output)
    y2_obs = y2_true + np.random.normal(0, noise_lvl, size=n_per_output)
    y3_obs = y3_true + np.random.normal(0, noise_lvl, size=n_per_output)
    
    Y_train = np.column_stack([y1_obs, y2_obs, y3_obs])
    X_test = np.linspace(0, 10, 50).reshape(-1, 1)

    print("Data generated. Outputs sum (approx):", np.max(np.abs(np.sum(Y_train, axis=1))))

    # 2. Setup Model with Constraint
    print("\n--- Model Setup ---")
    u_vector = np.array([1.0, 1.0, 1.0])
    print(f"Constraint vector u: {u_vector}")
    
    # 3 RBF Kernels
    k1 = RBFKernel(input_dim=1)
    k2 = RBFKernel(input_dim=1)
    k3 = RBFKernel(input_dim=1)
    
    kernel = LMCKernelConstrained(
        base_kernels=[k1, k2, k3], 
        output_dim=3, 
        u_vector=u_vector,
        rank=[1, 1, 1],
        seed=42
    )
    
    # Note: efficient likelihood might not be supported if we have multiple base kernels?
    # Actually LMCKernelConstrained has list of base kernels. If len(base_kernels) > 1, efficient lik is likely not used 
    # based on checking len(base_kernels)==1 in core.py. 
    # Here len=3. So naive likelihood will be used.
    
    model = MOGPR(kernel=kernel, use_efficient_lik=False, verbose=False)
    
    # 3. Fit
    print("Fitting model...")
    # use_init_pca is for LMCKernel, Constrained might handle init differently or not support it fully yet 
    # (init_L_from_pca is implemented in Constrained kernel to project init onto constraint)
    model.fit(X_train, Y_train, n_restarts=5, seed=42, use_init_pca=True)
    
    # 4. Verify Constraint on Matrix B
    print("\n--- Verifying Constraints on B ---")
    
    for q in range(3):
        B_q = kernel.get_B(q)
        print(f"Kernel {q} B matrix:\n{B_q}")
        
        # Check u.T @ B_q @ u = 0 (approx)
        val = u_vector @ B_q @ u_vector
        print(f"u.T @ B_{q} @ u = {val:.4e}")
        
        if abs(val) < 1e-9:
            print(f"Constraint satisfied for kernel {q}")
        else:
            print(f"Constraint VIOLATED for kernel {q}")

    # 5. Predict
    print("\nPredicting...")
    y_pred, _ = model.predict(X_test)
    
    # Check if predictions satisfy constraint approx
    pred_sum = np.sum(y_pred, axis=1)
    max_dev = np.max(np.abs(pred_sum))
    print(f"Max deviation from sum=0 in predictions: {max_dev:.4e}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_pred[:, 0], label='Pred y1')
    plt.plot(X_test, y_pred[:, 1], label='Pred y2')
    plt.plot(X_test, y_pred[:, 2], label='Pred y3')
    plt.plot(X_test, pred_sum, 'k--', label='Sum (Constraint)')
    plt.legend()
    plt.title("Constrained MOGP Predictions (y1+y2+y3=0)")
    plt.savefig('constrained_kernel_test.png')
    plt.show()
    print("Plot saved to constrained_kernel_test.png")

if __name__ == "__main__":
    main()
