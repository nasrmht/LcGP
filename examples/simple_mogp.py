import numpy as np
import matplotlib.pyplot as plt
from LcGP.mogp.core import MOGPR
from LcGP.mogp.kernels.LCMKernel import LMCKernel
from LcGP.mogp.kernels.Kernel import RBFKernel

def main():
    # 1. Generate synthetic Multi-Output data
    # Two outputs: y1 = sin(x), y2 = cos(x)
    np.random.seed(42)
    n_per_output = 20
    X_train = np.random.uniform(0, 10, size=(n_per_output, 1))
    
    y1 = np.sin(X_train).flatten() + np.random.normal(0, 0.05, size=n_per_output)
    y2 = np.cos(X_train).flatten() + np.random.normal(0, 0.05, size=n_per_output)
    
    # Combine outputs: MOGPR expects y of shape (n_samples, output_dim)
    Y_train = np.column_stack([y1, y2])
    
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y1_true = np.sin(X_test).flatten()
    y2_true = np.cos(X_test).flatten()
    
    # 2. Initialize Kernel and Model
    # LcGP.mogp.kernels.Kernel.RBFKernel takes input_dim
    rbf = RBFKernel(input_dim=1)
    
    # LMC Kernel with rank 2 for 2 outputs (full rank)
    kernel = LMCKernel(base_kernels=[rbf], output_dim=2, rank=[2])
    
    model = MOGPR(kernel=kernel, use_efficient_lik=False)
    
    # 3. Fit model
    print("Fitting MOGP model...")
    model.fit(X_train, Y_train, n_restarts=3, verbose=False)
    
    # 4. Predict
    print("Predicting...")
    y_pred, y_var = model.predict(X_test, return_cov=True)
    print("y_var shape :", y_var)
    
    y1_pred = y_pred[:, 0]
    y2_pred = y_pred[:, 1]
    
    y1_std = np.sqrt(y_var[:, 0])
    y2_std = np.sqrt(y_var[:, 1])
    
    # 5. Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Output 1
    axes[0].plot(X_test, y1_true, 'k--', label='True sin(x)')
    axes[0].scatter(X_train, y1, c='r', marker='x', label='Data y1')
    axes[0].plot(X_test, y1_pred, 'b-', label='Prediction')
    axes[0].fill_between(X_test.flatten(), y1_pred - 2*y1_std, y1_pred + 2*y1_std, color='b', alpha=0.2)
    axes[0].set_ylabel('y1')
    axes[0].set_title('Output 1')
    axes[0].legend()
    axes[0].grid(True)
    
    # Output 2
    axes[1].plot(X_test, y2_true, 'k--', label='True cos(x)')
    axes[1].scatter(X_train, y2, c='r', marker='x', label='Data y2')
    axes[1].plot(X_test, y2_pred, 'g-', label='Prediction')
    axes[1].fill_between(X_test.flatten(), y2_pred - 2*y2_std, y2_pred + 2*y2_std, color='g', alpha=0.2)
    axes[1].set_ylabel('y2')
    axes[1].set_xlabel('x')
    axes[1].set_title('Output 2')
    axes[1].legend()
    axes[1].grid(True)
    
    # Save plot
    plt.savefig('mogp_example.png')
    print("Plot saved to mogp_example.png")
    plt.show()

if __name__ == "__main__":
    main()
