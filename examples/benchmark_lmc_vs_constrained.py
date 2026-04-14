"""Benchmark LMCKernel vs LMCKernelConstrained on a 3-output toy problem.

Functions (normalized to mean~0, std~1 over [0,1]^2)
------------------------------------------------------
    f1(x) = Ishigami 2D (x3=0), normalisée
    f2(x) = Branin, normalisée
    f3(x) = -f1(x) - f2(x)    →  f1 + f2 + f3 = 0  exactement

Comparaisons reportées
-----------------------
    * Satisfaction de contrainte : max |f1_pred + f2_pred + f3_pred|
    * Q2 par sortie (train et test)
    * Vérification algébrique u.T B_q u = 0 pour le noyau contraint

Usage
-----
    cd LcGP
    python examples/benchmark_lmc_vs_constrained.py
"""
import numpy as np
from scipy.stats.qmc import LatinHypercube
from LcGP.mogp.core import MOGPR
from LcGP.mogp.kernels.LMCKernel import LMCKernel
from LcGP.mogp.kernels.ConstrainedLMCKernel import LMCKernelConstrained
from LcGP.mogp.kernels.Kernel import Matern52Kernel


# ------------------------------------------------------------------
# Normalization constants (analytical for Ishigami, empirical for Branin)
# ------------------------------------------------------------------
# Ishigami 2D (a=7, x3=0, x ∈ [0,1]² → [-π,π]²)
#   E[f1] = E[sin(x1)] + 7*E[sin²(x2)] = 0 + 7*0.5 = 3.5
#   Var[f1] = 0.5 + 49*(3/8 - 1/4) = 0.5 + 49/8 ≈ 6.625
_ISHI_MEAN = 3.5
_ISHI_STD  = np.sqrt(0.5 + 49.0 / 8.0)   # ≈ 2.574

# Branin on [0,1]² → x1'∈[-5,10], x2'∈[0,15]  (computed on a 10 000-point LHS grid)
_BRAN_MEAN = 53.95
_BRAN_STD  = 50.74


# ------------------------------------------------------------------
# Test functions
# ------------------------------------------------------------------

def ishigami_2d(x: np.ndarray) -> np.ndarray:
    """Ishigami (x3=0): sin(x1') + 7·sin²(x2'),  x' = 2π·x - π."""
    x1 = 2 * np.pi * x[:, 0] - np.pi
    x2 = 2 * np.pi * x[:, 1] - np.pi
    return np.sin(x1) + 7.0 * np.sin(x2) ** 2


def branin(x: np.ndarray) -> np.ndarray:
    """Branin, x ∈ [0,1]² → x1'∈[-5,10], x2'∈[0,15]."""
    a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi, 6.0, 10.0, 1.0 / (8 * np.pi)
    x1 = 15.0 * x[:, 0] - 5.0
    x2 = 15.0 * x[:, 1]
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1.0 - t) * np.cos(x1) + s


def f1_norm(x: np.ndarray) -> np.ndarray:
    """Ishigami 2D normalisée (mean~0, std~1)."""
    return (ishigami_2d(x) - _ISHI_MEAN) / _ISHI_STD


def f2_norm(x: np.ndarray) -> np.ndarray:
    """Branin normalisée (mean~0, std~1)."""
    return (branin(x) - _BRAN_MEAN) / _BRAN_STD


def generate_data(n: int, seed: int = 42):
    """Return (X (N,2), F (N,3)) with f1_norm + f2_norm + f3_norm = 0 exactement."""
    lhs = LatinHypercube(d=2, seed=seed)
    X = lhs.random(n)
    f1 = f1_norm(X)
    f2 = f2_norm(X)
    f3 = -f1 - f2
    return X, np.column_stack([f1, f2, f3])


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def q2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-15)


def constraint_error(Y_pred: np.ndarray, u: np.ndarray) -> dict:
    v = np.abs(Y_pred @ u)
    return {"max": v.max(), "mean": v.mean()}


# ------------------------------------------------------------------
# Model factories
# ------------------------------------------------------------------

def make_lmc(input_dim: int, output_dim: int, n_kernels: int, rank: int,
             seed: int = 42) -> LMCKernel:
    bk = [Matern52Kernel(input_dim=input_dim) for _ in range(n_kernels)]
    return LMCKernel(base_kernels=bk, output_dim=output_dim,
                     rank=[rank] * n_kernels, seed=seed)


def make_constrained(input_dim: int, output_dim: int, u: np.ndarray,
                     n_kernels: int, rank: int, seed: int = 42) -> LMCKernelConstrained:
    bk = [Matern52Kernel(input_dim=input_dim) for _ in range(n_kernels)]
    return LMCKernelConstrained(base_kernels=bk, output_dim=output_dim,
                                u_vector=u, rank=[rank] * n_kernels, seed=seed)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    np.random.seed(42)

    N_TRAIN    = 20
    N_TEST     = 1000
    N_KERNELS  = 3
    RANK       = 1
    N_RESTARTS = 5
    MAXITER    = 300
    NOISE_STD  = 0.0    # bruit relatif (fraction de l'écart-type par sortie)
    U = np.array([1.0, 1.0, 1.0])

    # ------------------------------------------------------------------
    # Données
    # ------------------------------------------------------------------
    X_train, F_train = generate_data(N_TRAIN, seed=42)
    X_test,  F_test  = generate_data(N_TEST,  seed=99)

    noise = NOISE_STD * F_train.std(axis=0)
    Y_train = F_train + np.random.randn(*F_train.shape) * noise[np.newaxis, :]

    print("=" * 65)
    print("Benchmark : LMCKernel vs LMCKernelConstrained")
    print("=" * 65)
    print(f"  N_train={N_TRAIN}, N_test={N_TEST}")
    print(f"  n_kernels={N_KERNELS}, rank={RANK}, n_restarts={N_RESTARTS}")
    print(f"  Contrainte : f1 + f2 + f3 = 0  (u = [1, 1, 1])")
    print(f"  Fonctions  : f1=Ishigami2D (normalisée), f2=Branin (normalisée), f3=-f1-f2")

    f_scales = F_train.std(axis=0)
    print(f"\n  Écart-types des sorties : f1={f_scales[0]:.3f}  "
          f"f2={f_scales[1]:.3f}  f3={f_scales[2]:.3f}")
    print(f"  Violation contrainte données d'entraînement (bruit seul) : "
          f"{np.abs(Y_train @ U).max():.2e}")

    # ------------------------------------------------------------------
    # Ajustement des deux modèles
    # ------------------------------------------------------------------
    results = {}

    for name, kernel in [
        ("LMCKernel",            make_lmc(2, 3, N_KERNELS, RANK)),
        ("LMCKernelConstrained", make_constrained(2, 3, U, N_KERNELS, RANK)),
    ]:
        print(f"\n--- Ajustement : {name} ---")
        model = MOGPR(kernel=kernel, noise_variance=1e-2,
                      use_efficient_lik=False, verbose=False)
        model.fit(X_train, Y_train, n_restarts=N_RESTARTS,
                  maxiter=MAXITER, use_init_pca=True)

        Y_pred_tr, _ = model.predict(X_train)
        Y_pred_te, _ = model.predict(X_test)

        results[name] = {
            "model":       model,
            "q2_train":    q2(Y_train,  Y_pred_tr),
            "q2_test":     q2(F_test,   Y_pred_te),
            "cstr_train":  constraint_error(Y_pred_tr, U),
            "cstr_test":   constraint_error(Y_pred_te, U),
        }

    # ------------------------------------------------------------------
    # Tableau de résultats
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("RÉSULTATS")
    print("=" * 65)

    labels = ["f1 (Ishigami)", "f2 (Branin)", "f3 = -f1-f2"]

    # Q2 test
    print(f"\nQ2 (test) :")
    print(f"  {'Modèle':<26}  {'f1':>8}  {'f2':>8}  {'f3':>8}  {'moy':>8}")
    print("  " + "-" * 50)
    for name, r in results.items():
        v = r["q2_test"]
        print(f"  {name:<26}  {v[0]:>+8.4f}  {v[1]:>+8.4f}  {v[2]:>+8.4f}  {v.mean():>+8.4f}")

    # Satisfaction contrainte
    print(f"\nSatisfaction contrainte  |f1_pred + f2_pred + f3_pred| :")
    print(f"  {'Modèle':<26}  {'train max':>10}  {'test max':>10}  {'test mean':>10}")
    print("  " + "-" * 60)
    for name, r in results.items():
        tr = r["cstr_train"]
        te = r["cstr_test"]
        print(f"  {name:<26}  {tr['max']:>10.2e}  {te['max']:>10.2e}  {te['mean']:>10.2e}")

    # Amélioration
    lmc_max  = results["LMCKernel"]["cstr_test"]["max"]
    cst_max  = results["LMCKernelConstrained"]["cstr_test"]["max"]
    ratio    = lmc_max / (cst_max + 1e-20)
    print(f"\n  Amélioration contrainte  LMCKernel / LMCKernelConstrained : {ratio:.1e}×")

    # ------------------------------------------------------------------
    # Vérification algébrique u.T B_q u = 0 (noyau contraint)
    # ------------------------------------------------------------------
    print("\n--- Vérification algébrique u.T B_q u (LMCKernelConstrained) ---")
    kern_c = results["LMCKernelConstrained"]["model"].kernel
    for q in range(N_KERNELS):
        B_q = kern_c.get_B(q)
        val = U @ B_q @ U
        tag = "OK" if abs(val) < 1e-10 else "VIOLATION"
        print(f"  B_{q} : u.T B_q u = {val:.2e}  [{tag}]")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
