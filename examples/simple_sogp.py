import numpy as np
import matplotlib.pyplot as plt
from LcGP.sogp.core import so_GPRegression
from LcGP.sogp.kernels.Kernel import RBFKernel

def main():
    # 1. Generate synthetic data
    np.random.seed(42)
    X_train = np.random.uniform(0, 10, size=(20, 1))
    y_train = np.sin(X_train).flatten() + np.random.normal(0, 0.1, size=20)
    
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_true = np.sin(X_test).flatten()
    
    # 2. Initialize Kernel and Model
    # Note: LcGP.sogp.kernels.Kernel.RBFKernel takes length_scale, not input_dim
    kernel = RBFKernel(length_scale=1.0)
    model = so_GPRegression(kernel=kernel, noisy_data=True, var_noise=0.01)
    
    # 3. Fit model
    print("Fitting SoGP model...")
    model.fit(X_train, y_train, multi_start=True, n_start=5)
    print(f"Optimized hyperparameters: {model.hyperparameters}")
    
    # 4. Predict
    print("Predicting...")
    y_pred, y_var = model.predict(X_test)
    y_std = np.sqrt(y_var)
    
    # 5. Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_true, 'k--', label='True function')
    plt.scatter(X_train, y_train, c='r', marker='x', label='Training data')
    plt.plot(X_test, y_pred, 'b-', label='Prediction')
    plt.fill_between(X_test.flatten(), y_pred - 2*y_std, y_pred + 2*y_std, color='b', alpha=0.2, label='95% CI')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Single-Output Gaussian Process Regression (LcGP)')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig('sogp_example.png')
    print("Plot saved to sogp_example.png")
    plt.show()

if __name__ == "__main__":
    main()
