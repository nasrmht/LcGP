import numpy as np
from scipy import linalg
from typing import List, Tuple, Dict, Optional, Union, Callable
import time
import sys
import os

from Multi_output_GP.MOGPLikelihood.log_likelihood_naive import compute_log_likelihood_naive, compute_log_likelihood_gradient_naive
#from ..MOGPKernel.LCMKernel import LMCKernel


# Ajouter le répertoire parent au chemin Python pour permettre les importations
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Multi_output_GP.MOGPKernel.LCMKernel import LMCKernel
#from MOGPKernel.Kernel import Kernel

def compute_log_likelihood_kronecker(kernel, X: np.ndarray, y: np.ndarray, log_noise_variance: float) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calcule la log-vraisemblance en utilisant la structure de Kronecker pour un modèle LCM.
    
    Cette implémentation est optimisée pour exploiter la structure spécifique du modèle LCM
    où le noyau est défini comme une somme de produits de Kronecker.
    
    Basée sur l'équation:
    logp(V|A, R_j=1^p) = np log(2π)/2 + n log |det(A)| + ∑_j=1^p log det(R_j) - (1/2) ∑_j=1^p a_j^-1 V R_j^-1 V^T (a_j^-1)^T
    
    Args:
        kernel: Instance de LMCKernel
        X: Matrice d'entrée préparée de forme (n * output_dim, input_dim + 1)
        y: Vecteur de sortie préparé de forme (n * output_dim,)
        log_noise_variance: Logarithme de la variance du bruit
        
    Returns:
        Tuple contenant:
            - La log-vraisemblance négative (pour la minimisation)
            - Le facteur de Cholesky L
            - La solution alpha du système linéaire
    """
    # Récupérer les dimensions
    input_dim = X.shape[1] - 1  # La dernière colonne contient l'indice de sortie
    output_dim = kernel.output_dim
    n_samples = X.shape[0] // output_dim  # Nombre d'échantillons spatiaux
    
    # Extraire les entrées spatiales et indices de sortie
    X_spatial = X[:, :-1].reshape(n_samples, output_dim, input_dim)[:, 0, :]  # Prendre les coordonnées spatiales uniques
    output_idx = X[:, -1].astype(int)
    
    # Organiser y en matrice Y de forme (n_samples, output_dim)
    Y = np.zeros((n_samples, output_dim))
    for i in range(X.shape[0]):
        # Trouver l'indice spatial et l'indice de sortie
        spatial_idx = i // output_dim
        output_idx_i = output_idx[i]
        Y[spatial_idx, output_idx_i] = y[i]
    
    # Considérer Y comme une matrice de forme (n_samples, output_dim)
    # où chaque ligne correspond à un point spatial et chaque colonne à une sortie
    
    # Transformer Y en matrice V de forme (output_dim, n_samples) pour correspondre à la notation de l'équation
    V = Y.T
    
    # Calculer les composantes individuelles du modèle LCM
    noise_variance = np.exp(log_noise_variance)
    
    # Construire la matrice A (dans le contexte du LCM, A est liée aux matrices B_q)
    # Pour un LCM, la matrice A peut être construite à partir des matrices L_q
    # Ici, nous utilisons les matrices L_q du noyau LMC existant
    
    # Pas 1: Calculer les matrices de covariance spatiale R_j pour chaque noyau de base
    R_matrices = []
    for q, base_kernel in enumerate(kernel.base_kernels):
        R_q = base_kernel(X_spatial)
        
        # Ajouter le bruit à la diagonale uniquement pour le calcul numérique stable
        if q == 0:  # Ajouter le bruit seulement à la première matrice
            R_q += noise_variance * np.eye(n_samples)
        
        R_matrices.append(R_q)
    
    # Pas 2: Récupérer les matrices L_q du noyau LMC
    L_matrices = [kernel.get_L(q) for q in range(len(kernel.base_kernels))]
    
    # Construire la matrice A en concaténant les matrices L_q
    A = np.column_stack([L_q * np.sqrt(kernel.get_sigma_B(q)) for q, L_q in enumerate(L_matrices)])
    
    # Calculer A_inv (A inverse)
    A_inv = np.linalg.inv(A)
    
    # Initialiser la log-vraisemblance
    log_likelihood = 0.0
    
    # Terme 1: np log(2π)/2
    log_likelihood += -0.5 * n_samples * output_dim * np.log(2 * np.pi)
    
    # Terme 2: n log |det(A)|
    log_det_A = np.log(np.abs(np.linalg.det(A)))
    log_likelihood += n_samples * log_det_A
    
    # Termes 3 et 4: ∑_j log det(R_j) - 0.5 * ∑_j a_j^-1 V R_j^-1 V^T (a_j^-1)^T
    p = len(kernel.base_kernels)
    for j in range(p):
        # Calculer log det(R_j)
        sign, logdet_R_j = np.linalg.slogdet(R_matrices[j])
        log_likelihood += logdet_R_j
        
        # Calculer R_j^-1
        R_j_inv = np.linalg.inv(R_matrices[j])
        
        # Extraire a_j^-1 (j-ième colonne de A_inv)
        a_j_inv = A_inv[:, j].reshape(-1, 1)
        
        # Calculer a_j^-1 V
        a_j_inv_V = a_j_inv.T @ V
        
        # Calculer a_j^-1 V R_j^-1
        a_j_inv_V_R_j_inv = a_j_inv_V @ R_j_inv
        
        # Calculer a_j^-1 V R_j^-1 V^T (a_j^-1)^T
        quadratic_term = a_j_inv_V_R_j_inv @ V.T @ a_j_inv
        
        # Soustraire le terme quadratique
        log_likelihood -= 0.5 * quadratic_term[0, 0]
    
    # Pour la compatibilité avec l'interface existante, calculons équivalents de L et alpha
    # Même si ce ne sont pas utilisés directement dans cette implémentation
    
    # Pour L, on peut utiliser une approximation pour maintenir la compatibilité
    K_approx = np.zeros((X.shape[0], X.shape[0]))
    L = np.eye(X.shape[0])  # Facteur de Cholesky approximatif
    alpha = np.zeros_like(y)  # Solution alpha approximative
    
    return -log_likelihood, L, alpha

def compute_log_likelihood_gradient_kronecker(kernel, X: np.ndarray, y: np.ndarray, log_noise_variance: float):
    """
    Calcule le gradient de la log-vraisemblance optimisée pour un modèle LCM 
    en utilisant la structure de Kronecker.
    
    Cette version est un calcul approximatif du gradient qui fonctionne avec l'optimisation
    par différences finies. Pour une implémentation complète du gradient analytique, 
    il faudrait dériver chaque terme par rapport à chaque paramètre.
    
    Args:
        kernel: Instance de LMCKernel
        X: Matrice d'entrée préparée
        y: Vecteur de sortie préparé
        log_noise_variance: Logarithme de la variance du bruit
        
    Returns:
        Gradient de la log-vraisemblance négative par rapport aux paramètres
    """
    # Utiliser une différenciation numérique pour le gradient
    h = 1e-6  # Pas de différenciation
    
    # Calculer la log-vraisemblance initiale
    ll_current, _, _ = compute_log_likelihood_kronecker(kernel, X, y, log_noise_variance)
    
    # Initialiser le gradient
    grad = np.zeros(len(kernel.params) + 1)
    
    # Calculer le gradient pour chaque paramètre du noyau
    for i in range(len(kernel.params)):
        # Sauvegarder la valeur actuelle du paramètre
        theta_current = kernel.params[i]
        
        # Perturber le paramètre
        params_perturbed = kernel.params.copy()
        params_perturbed[i] += h
        kernel.params = params_perturbed
        
        # Calculer la log-vraisemblance avec le paramètre perturbé
        ll_perturbed, _, _ = compute_log_likelihood_kronecker(kernel, X, y, log_noise_variance)
        
        # Calculer la dérivée partielle
        grad[i] = (ll_perturbed - ll_current) / h
        
        # Restaurer la valeur originale du paramètre
        params_original = kernel.params.copy()
        params_original[i] = theta_current
        kernel.params = params_original
    
    # Calculer le gradient pour le paramètre de bruit
    noise_perturbed = log_noise_variance + h
    ll_perturbed, _, _ = compute_log_likelihood_kronecker(kernel, X, y, noise_perturbed)
    grad[-1] = (ll_perturbed - ll_current) / h
    
    return grad

# Fonction d'optimisation avec comparaison des performances
def optimize_with_performance_comparison(kernel, X, y, log_noise_variance, n_iter=100):
    """
    Optimise les paramètres du modèle en utilisant à la fois la méthode naïve 
    et la méthode optimisée pour comparer les performances.
    
    Args:
        kernel: Instance de LMCKernel
        X: Matrice d'entrée préparée
        y: Vecteur de sortie préparé
        log_noise_variance: Logarithme de la variance du bruit initial
        n_iter: Nombre d'itérations pour l'optimisation
        
    Returns:
        Dictionnaire contenant les résultats de l'optimisation et les métriques de performance
    """
    from scipy.optimize import minimize
    
    # Fonction objectif naïve
    def objective_naive(params):
        kernel_params = params[:-1]
        noise_param = params[-1]
        
        kernel.params = kernel_params
        ll, _, _ = compute_log_likelihood_naive(kernel, X, y, noise_param)
        return ll
    
    # Fonction objectif optimisée
    def objective_kronecker(params):
        kernel_params = params[:-1]
        noise_param = params[-1]
        
        kernel.params = kernel_params
        ll, _, _ = compute_log_likelihood_kronecker(kernel, X, y, noise_param)
        return ll
    
    # Paramètres initiaux
    initial_params = np.concatenate([kernel.params, [log_noise_variance]])
    
    # Mesurer le temps pour la méthode naïve
    start_time = time.time()
    result_naive = minimize(
        objective_naive,
        initial_params,
        method='L-BFGS-B',
        options={'maxiter': n_iter}
    )
    naive_time = time.time() - start_time
    
    # Réinitialiser les paramètres
    kernel.params = initial_params[:-1]
    
    # Mesurer le temps pour la méthode optimisée
    start_time = time.time()
    result_kronecker = minimize(
        objective_kronecker,
        initial_params,
        method='L-BFGS-B',
        options={'maxiter': n_iter}
    )
    kronecker_time = time.time() - start_time
    
    # Calculer le gain de performance
    speedup = naive_time / kronecker_time
    
    return {
        'naive_result': result_naive,
        'kronecker_result': result_kronecker,
        'naive_time': naive_time,
        'kronecker_time': kronecker_time,
        'speedup': speedup
    }

# Exemple d'utilisation
def example_usage():
    """
    Exemple d'utilisation des fonctions optimisées pour un cas synthétique.
    """
    from Multi_output_GP.MOGPKernel.Kernel import Kernel
    
    # Créer des données synthétiques
    n_samples = 100
    input_dim = 2
    output_dim = 3
    
    # Créer des noyaux de base
    class SimpleRBF(Kernel):
        def __init__(self, lengthscale=1.0, variance=1.0):
            self.lengthscale = lengthscale
            self.variance = variance
        
        @property
        def params(self):
            return np.array([self.lengthscale, self.variance])
        
        @params.setter
        def params(self, params):
            self.lengthscale = params[0]
            self.variance = params[1]
        
        def __call__(self, X1, X2=None):
            if X2 is None:
                X2 = X1
            
            # Calcul de la distance euclidienne
            X1 = X1 / self.lengthscale
            X2 = X2 / self.lengthscale
            
            n1, n2 = X1.shape[0], X2.shape[0]
            dist = np.zeros((n1, n2))
            
            for i in range(n1):
                for j in range(n2):
                    dist[i, j] = np.sum((X1[i] - X2[j])**2)
            
            return self.variance * np.exp(-0.5 * dist)
        
        def gradient(self, X1, X2=None):
            if X2 is None:
                X2 = X1
            
            K = self(X1, X2)
            n1, n2 = X1.shape[0], X2.shape[0]
            
            dK_dl = np.zeros((n1, n2))
            dK_dv = np.zeros((n1, n2))
            
            X1 = X1 / self.lengthscale
            X2 = X2 / self.lengthscale
            
            for i in range(n1):
                for j in range(n2):
                    dist_sq = np.sum((X1[i] - X2[j])**2)
                    dK_dl[i, j] = K[i, j] * dist_sq / self.lengthscale
                    dK_dv[i, j] = K[i, j] / self.variance
            
            return [dK_dl, dK_dv]
        
        @property
        def bounds(self):
            return [(1e-6, 10.0), (1e-6, 10.0)]
        
        def get_n_params(self):
            return 2
    
    # Créer un noyau LMC
    kernel_list = [SimpleRBF(1.0, 1.0), SimpleRBF(0.5, 2.0)]
    lmc_kernel = LMCKernel(kernel_list, output_dim=output_dim)
    
    # Générer des entrées
    X_spatial = np.random.rand(n_samples, input_dim)
    
    # Créer la matrice X étendue pour le MOGP
    X = np.zeros((n_samples * output_dim, input_dim + 1))
    for i in range(n_samples):
        for d in range(output_dim):
            idx = i * output_dim + d
            X[idx, :-1] = X_spatial[i]
            X[idx, -1] = d
    
    # Générer des sorties
    K = lmc_kernel(X)
    y = np.random.multivariate_normal(np.zeros(n_samples * output_dim), K + 1e-6 * np.eye(n_samples * output_dim))
    
    # Calculer la log-vraisemblance avec les deux méthodes
    log_noise = np.log(1e-3)
    
    start = time.time()
    ll_naive, _, _ = compute_log_likelihood_naive(lmc_kernel, X, y, log_noise)
    time_naive = time.time() - start
    
    start = time.time()
    ll_optimized, _, _ = compute_log_likelihood_kronecker(lmc_kernel, X, y, log_noise)
    time_optimized = time.time() - start
    
    print(f"Log-vraisemblance naïve: {ll_naive:.4f}, temps: {time_naive:.4f} sec")
    print(f"Log-vraisemblance optimisée: {ll_optimized:.4f}, temps: {time_optimized:.4f} sec")
    print(f"Accélération: {time_naive / time_optimized:.2f}x")
    
    # Optimiser les paramètres
    results = optimize_with_performance_comparison(lmc_kernel, X, y, log_noise, n_iter=10)
    
    print("\nRésultats d'optimisation:")
    print(f"Méthode naïve - temps: {results['naive_time']:.4f} sec")
    print(f"Méthode optimisée - temps: {results['kronecker_time']:.4f} sec")
    print(f"Gain de performance: {results['speedup']:.2f}x")
    
    return results

if __name__ == "__main__":
    example_usage()