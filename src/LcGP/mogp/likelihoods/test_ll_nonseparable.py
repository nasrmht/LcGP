import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, solve, det
import matplotlib.pyplot as plt
import time

# --- Corrélation exponentielle ---
def exp_corr(loc1, loc2=None, phi=1.0):
    if loc2 is None:
        loc2 = loc1
    D = cdist(loc1, loc2)
    return np.exp(-phi * D)

# --- Simulation LMC non-séparable ---
def simulate_lmc(A, phis, locations):
    p = A.shape[0]
    n = locations.shape[0]
    W = np.zeros((p, n))
    for j in range(p):
        Rj = exp_corr(locations, phi=phis[j])
        Lj = cholesky(Rj, lower=True)
        W[j, :] = Lj @ np.random.randn(n)
    V = A @ W
    return V

# --- Vraisemblance naïve ---
def loglik_naive(V, A, phis, locations):
    p, n = V.shape
    Sigma = np.zeros((p*n, p*n))
    # t1 = time.time()
    for j in range(p):
        Rj = exp_corr(locations, phi=phis[j])
        aj = A[:, j][:, None]
        Sigma += np.kron(Rj, aj @ aj.T)
    L = cholesky(Sigma, lower=True)
    alpha_y = solve(L, V.T.flatten())  #solve(L.T, solve(L, V.T.flatten()))
    logdet = 2 * np.sum(np.log(np.diag(L)))
    # print("t naif = ", time.time()-t1)
    return -0.5 * (V.size*np.log(2*np.pi) + logdet +np.sum(alpha_y**2) ) #V.T.flatten() @ alpha)

# --- Vraisemblance efficace ---
def loglik_efficient(V, A, phis, locations):
    p, n = V.shape
    Ainv = np.linalg.inv(A)
    total_quad = 0.0
    logdetR_sum = 0.0
    #t1 = time.time()
    for j in range(p):
        Rj = exp_corr(locations, phi=phis[j])
        Lj = cholesky(Rj, lower=True)
        logdetR_sum += 2 * np.sum(np.log(np.diag(Lj)))
        yj = Ainv[j, :] @ V  # ligne j de W
        alpha_j = solve(Lj, yj)  #solve(Lj.T, solve(Lj, yj))
        total_quad += np.sum(alpha_j**2) #yj @ alpha_j
    loglik = -0.5 * (p*n*np.log(2*np.pi) + 2*n*np.log(abs(det(A))) + logdetR_sum + total_quad)
    #print("t eff = ", time.time()-t1)
    return loglik

# --- Prédiction naïve ---
def lmc_predict_naive(A, phis, loc_obs, loc_pred, V_obs):
    p, n_obs = V_obs.shape
    n_pred = loc_pred.shape[0]
    # Construire la covariance complète
    Sigma_oo = np.zeros((p*n_obs, p*n_obs))
    Sigma_po = np.zeros((p*n_pred, p*n_obs))
    Sigma_pp = np.zeros((p*n_pred, p*n_pred))
    for j in range(p):
        R_oo = exp_corr(loc_obs, phi=phis[j])
        R_po = exp_corr(loc_pred, loc_obs, phi=phis[j])
        R_pp = exp_corr(loc_pred, phi=phis[j])
        aj = A[:, j][:, None]
        Sigma_oo += np.kron(R_oo, aj @ aj.T)
        Sigma_po += np.kron(R_po, aj @ aj.T)
        Sigma_pp += np.kron(R_pp, aj @ aj.T)
    # Moyenne et covariance conditionnelle
    L_oo = cholesky(Sigma_oo, lower=True)
    alpha = solve(L_oo.T, solve(L_oo, V_obs.T.flatten()))
    mean_pred = Sigma_po @ alpha
    cov_pred = Sigma_pp - Sigma_po @ solve(Sigma_oo, Sigma_po.T)
    return mean_pred.reshape(p, n_pred)

# --- Prédiction efficace ---
def lmc_predict_efficient(A, phis, loc_obs, loc_pred, V_obs):
    p, n_obs = V_obs.shape
    n_pred = loc_pred.shape[0]
    Ainv = np.linalg.inv(A)
    W_obs = Ainv @ V_obs
    W_pred_mean = np.zeros((p, n_pred))
    for j in range(p):
        R_oo = exp_corr(loc_obs, phi=phis[j])
        R_po = exp_corr(loc_pred, loc_obs, phi=phis[j])
        alpha = solve(R_oo, W_obs[j, :])
        W_pred_mean[j, :] = R_po @ alpha
    V_pred_mean = A @ W_pred_mean
    return V_pred_mean

# --- RMSE et Q2 ---
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def q2(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

# --- Comparaison précision ---
def compare_prediction_accuracy():
    np.random.seed(0)
    n_obs, n_pred, p = 80, 20, 3
    loc_obs = np.random.rand(n_obs, 2)
    loc_pred = np.random.rand(n_pred, 2)
    A = np.eye(p) + 0.2*np.random.randn(p, p)
    phis = np.linspace(5, 15, p)
    V_full = simulate_lmc(A, phis, np.vstack([loc_obs, loc_pred]))
    V_obs = V_full[:, :n_obs]
    V_true = V_full[:, n_obs:]

    # Prédictions
    V_pred_naive = lmc_predict_naive(A, phis, loc_obs, loc_pred, V_obs)
    V_pred_eff = lmc_predict_efficient(A, phis, loc_obs, loc_pred, V_obs)

    # Erreurs
    rmse_naive = rmse(V_true, V_pred_naive)
    rmse_eff = rmse(V_true, V_pred_eff)
    q2_naive = q2(V_true.flatten(), V_pred_naive.flatten())
    q2_eff = q2(V_true.flatten(), V_pred_eff.flatten())

    print(f"Naïve: RMSE={rmse_naive:.6f}, Q2={q2_naive:.6f}")
    print(f"Efficace: RMSE={rmse_eff:.6f}, Q2={q2_eff:.6f}")

# --- Benchmark temps ---
def benchmark_times(n=1000, p_values=[6, 10], reps=1):
    locs = np.random.rand(n, 2)
    times_naive = []
    times_eff = []
    for p in p_values:
        A = np.eye(p) + 0.2*np.random.randn(p, p)
        phis = np.linspace(5, 20, p)
        V = simulate_lmc(A, phis, locs)
        # Naive
        t0 = time.time()
        for _ in range(reps):
            loglik_naive(V, A, phis, locs)
        times_naive.append(time.time() - t0)
        print("log naive  =  ",loglik_naive(V, A, phis, locs))
        # Efficient
        t0 = time.time()
        for _ in range(reps):
            loglik_efficient(V, A, phis, locs)
        times_eff.append(time.time() - t0)
        print("log effi  =  ",loglik_efficient(V, A, phis, locs))
    return p_values, times_naive, times_eff

# --- Lancer benchmark ---
p_vals, t_naive, t_eff = benchmark_times()

# --- Graphique temps ---
plt.figure(figsize=(6,4))
plt.plot(p_vals, t_naive, label="Naïve")
plt.plot(p_vals, t_eff, label="Efficace")
plt.xlabel("Dimension p des sorties")
plt.ylabel("Temps total (500 rép.) [s]")
plt.title("Comparaison temps log-vraisemblance (n=100)")
plt.legend()
plt.grid(True)
plt.show()

# --- Comparaison RMSE/Q2 ---
compare_prediction_accuracy()
