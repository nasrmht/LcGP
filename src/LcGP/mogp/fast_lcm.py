# import numpy as np
# import scipy.linalg as spl
# from scipy.optimize import minimize
# from scipy.spatial.distance import pdist, squareform, cdist
# from scipy.stats import qmc
# from sklearn.decomposition import PCA
# from joblib import Parallel, delayed

# class FastLMCKernel:
#     """
#     Gère la structure des hyperparamètres :
#     - Sigma_A : Facteur d'échelle global de A [1, 10]
#     - A_norm : Matrice normalisée (tous les éléments) [-1, 1]
#     - Log_Lengthscales : Logarithme des portées (pour positivité garantie)
#     """
#     def __init__(self, n_features, p_components, kernel_type='rbf'):
#         self.d = n_features
#         self.p = p_components
#         self.kernel_type = kernel_type
        
#     def get_parameter_bounds(self, X_min, X_max):
#         """
#         Définit les bornes. Les lengthscales sont converties en log.
#         """
#         # 1. Borne pour sigma_A (Facteur d'échelle) : [1, 10]
#         bounds_sigma = [(1.0, 10.0)]
        
#         # 2. Bornes pour TOUS les éléments de A normalisés : [-1, 1]
#         # On optimise p*p éléments maintenant (plus de fixation à 1)
#         n_A_elems = self.p * self.p
#         bounds_A_norm = [(-1.0, 1.0)] * n_A_elems
        
#         # 3. Bornes pour les LOG-lengthscales
#         ranges = X_max - X_min
#         ranges[ranges == 0] = 1.0 
        
#         bounds_log_lengthscales = []
#         for _ in range(self.p): 
#             for dim in range(self.d):
#                 # Bornes physiques : 1e-3 à 10 * range
#                 lower_phys = 1e-3
#                 upper_phys = 1.0 * ranges[dim]
                
#                 # Conversion en bornes LOG
#                 bounds_log_lengthscales.append((np.log(lower_phys), np.log(upper_phys)))
                
#         return bounds_sigma + bounds_A_norm + bounds_log_lengthscales

#     def unpack_params(self, theta):
#         """
#         Reconstruit A et les lengthscales (en appliquant exp).
#         """
#         # 1. Extraction de sigma_A
#         sigma_A = theta[0]
        
#         # 2. Reconstruction de A
#         # On a p^2 éléments
#         n_A_elems = self.p * self.p
#         a_params = theta[1 : 1 + n_A_elems]
        
#         A_norm = a_params.reshape(self.p, self.p)
#         A = sigma_A * A_norm
        
#         # 3. Extraction des lengthscales (Log -> Exp)
#         log_ls_params = theta[1 + n_A_elems :]
#         # C'est ici que la transformation se fait
#         lengthscales = np.exp(log_ls_params).reshape(self.p, self.d)
        
#         return A, lengthscales

# class FastLMCNll:
#     """Calcul de la NLL O(p)"""
#     def __init__(self, kernel, sparsity_lambda=0.0):
#         self.kernel = kernel
#         self.lam = sparsity_lambda

#     def _compute_kernel_matrix(self, X, lengthscales_j):
#         # lengthscales_j est déjà exponentié par unpack_params
#         X_scaled = X / lengthscales_j
#         dists = pdist(X_scaled, metric='sqeuclidean')
        
#         if self.kernel.kernel_type == 'rbf':
#             K = np.exp(-0.5 * dists)
#         elif self.kernel.kernel_type == 'matern32':
#             d = np.sqrt(dists)
#             sqrt3 = np.sqrt(3.0)
#             K = (1.0 + sqrt3 * d) * np.exp(-sqrt3 * d)
#         return squareform(K)

#     def compute(self, theta, X, Y):
#         # 1. Déballage (Gère le log-transform)
#         try:
#             A, lengthscales = self.kernel.unpack_params(theta)
#         except Exception:
#             return 1e15
#         print("Lengthscales:", lengthscales)    
#         # 2. Inversibilité A
#         # try:
#         #     sign, logdet_A = np.linalg.slogdet(A)
#         #     if sign <= 0: return 1e15 # Déterminant doit être positif
#         #     inv_A = spl.inv(A)
#         # except np.linalg.LinAlgError:
#         #     return 1e15
#         sign, logdet_A = np.linalg.slogdet(A)
#         inv_A = spl.inv(A)
#         #print("inv_A computed. :", inv_A)
#         # 3. Projection Latente
#         W_hat = inv_A @ Y.T

#         # 4. Somme sur les processus latents
#         log_det_Rs = 0
#         quad_form_sum = 0
#         n = X.shape[0]
#         p = Y.shape[1]
        
#         for j in range(p):
#             R_j = self._compute_kernel_matrix(X, lengthscales[j, :])
#             # Jitter minimal pour l'optimisation
#             R_j[np.diag_indices_from(R_j)] += 1e-6 
            
#             try:
#                 L = spl.cholesky(R_j, lower=True)
#                 log_det_Rs += 2 * np.sum(np.log(np.diag(L)))
#                 w_j = W_hat[j, :]
#                 alpha = spl.cho_solve((L, True), w_j)
#                 quad_form_sum += w_j @ alpha
#             except np.linalg.LinAlgError:
#                 return 1e15

#         const = n * p * np.log(2 * np.pi)
#         total_log_det = 2 * n * logdet_A + log_det_Rs
#         nll = 0.5 * (const + total_log_det + quad_form_sum)
        
#         if self.lam > 0:
#             nll += self.lam * np.sum(np.abs(A))
#         return nll

# class FastSparseLMC:
#     def __init__(self, p_components, kernel_type='rbf', sparsity_lambda=0.0, n_restarts=5, use_pca_init=True, n_jobs=1, seed=42):
#         self.p = p_components
#         self.kernel_type = kernel_type
#         self.lam = sparsity_lambda
#         self.n_restarts = n_restarts
#         self.use_pca_init = use_pca_init
#         self.n_jobs = n_jobs
        
#         self.kernel = None 
#         self.best_params_ = None
#         self.seed = seed
#         # Cache
#         self.cached_inv_R_ = []
#         self.cached_alpha_ = []
#         self.A_hat_ = None
#         self.phis_hat_ = None

#     def _initialize_hyperparams(self, n_starts, bounds):
#         n_params = len(bounds)
#         lower_bounds = np.array([b[0] for b in bounds])
#         upper_bounds = np.array([b[1] for b in bounds])
        
#         sampler = qmc.LatinHypercube(d=n_params, optimization="random-cd", rng=self.seed)
#         sample = sampler.random(n=n_starts)
#         initial_thetas = qmc.scale(sample, lower_bounds, upper_bounds)
        
#         # --- INIT SPÉCIALE POUR SIGMA ---
#         # On force sigma à démarrer à 1.0 (ou proche) pour aider la convergence
#         # Index 0 est sigma
#         initial_thetas[:, 0] = 1.0 # + 0.1 * np.random.randn(n_starts)
#         initial_thetas[:, 0] = np.clip(initial_thetas[:, 0], 1.0, 10.0)

#         # --- INIT PCA ---
#         if self.use_pca_init and self.Y_train_ is not None:
#             try:
#                 pca = PCA(n_components=self.p)
#                 pca.fit(self.Y_train_)
#                 A_pca = pca.components_.T # (p, p)
                
#                 # Normalisation adaptée à la nouvelle structure
#                 max_val = np.max(np.abs(A_pca))
#                 if max_val < 1e-9: max_val = 1.0
                
#                 # On met l'échelle dans sigma (borné [1, 10])
#                 sigma_init = np.clip(max_val, 1.0, 10.0)
                
#                 # On normalise A entre -1 et 1
#                 A_norm_init = A_pca / max_val
#                 # Clip de sécurité
#                 A_norm_init = np.clip(A_norm_init, -1.0, 1.0)
                
#                 A_flat = A_norm_init.flatten()
                
#                 # On applique cette init de A à tous les starts LHS
#                 # Index 0 = sigma
#                 # Index 1 à 1+p^2 = A
#                 for i in range(n_starts):
#                     initial_thetas[i, 0] = sigma_init
#                     initial_thetas[i, 1 : 1 + len(A_flat)] = A_flat
                    
#             except Exception as e:
#                 print(f"Warning: PCA init failed ({e})")

#         return initial_thetas, list(zip(lower_bounds, upper_bounds))

#     def _optimize_hyperparams(self, initial_theta, bounds):
#         # Utilisation de L-BFGS-B qui gère bien les bornes
#         # try:
#         res = minimize(
#             self.nll_engine.compute,
#             initial_theta,
#             args=(self.X_train_, self.Y_train_),
#             method='L-BFGS-B',
#             bounds=bounds,
#             options={'maxiter': 200} #, 'ftol': 1e-9}
#         )
#         return res
#         # except Exception:
#         #     return None

#     def fit(self, X, Y):
#         self.X_train_ = X
#         self.Y_train_ = Y
#         n_features = X.shape[1]
        
#         self.kernel = FastLMCKernel(n_features, self.p, self.kernel_type)
#         self.nll_engine = FastLMCNll(self.kernel, self.lam)
        
#         # Bornes
#         X_min = np.min(X, axis=0)
#         X_max = np.max(X, axis=0)
#         bounds_def = self.kernel.get_parameter_bounds(X_min, X_max)
        
#         # Initialisation
#         starts, bounds = self._initialize_hyperparams(self.n_restarts, bounds_def)
        
#         # Optimisation parallèle
#         # results = Parallel(n_jobs=self.n_jobs)(
#         #     delayed(self._optimize_hyperparams)(theta, bounds) for theta in starts
#         # )
#         results=[]
#         for theta in starts:
#             res=self._optimize_hyperparams(theta, bounds)
#             results.append(res)
        
#         valid_results = [r for r in results if r is not None and r.success]
#         if not valid_results: valid_results = [r for r in results if r is not None]
#         if not valid_results: raise RuntimeError("Optim failed completely.")
            
#         best_res = min(valid_results, key=lambda x: x.fun)
#         self.best_params_ = best_res.x
#         self.final_nll_ = best_res.fun
        
#         # --- MISE EN CACHE ROBUSTE (Identique à avant) ---
#         self.A_hat_, self.phis_hat_ = self.kernel.unpack_params(self.best_params_)
        
#         try: inv_A = spl.inv(self.A_hat_)
#         except: inv_A = spl.pinv(self.A_hat_)
#         W_obs = inv_A @ self.Y_train_.T
        
#         self.cached_inv_R_ = []
#         self.cached_alpha_ = []
        
#         for j in range(self.p):
#             ls = self.phis_hat_[j, :]
#             X_scaled = self.X_train_ / ls
#             dists = squareform(pdist(X_scaled, metric='sqeuclidean'))
            
#             if self.kernel_type == 'rbf': K_base = np.exp(-0.5 * dists)
#             elif self.kernel_type == 'matern32':
#                 d = np.sqrt(dists); sqrt3 = np.sqrt(3.0)
#                 K_base = (1.0 + sqrt3 * d) * np.exp(-sqrt3 * d)
            
#             # Robust Cholesky (Jitter dynamique)
#             jitter = 1e-6
#             L = None
#             for _ in range(5):
#                 try:
#                     K_mat = K_base.copy()
#                     K_mat[np.diag_indices_from(K_mat)] += jitter
#                     L = spl.cholesky(K_mat, lower=True)
#                     break
#                 except np.linalg.LinAlgError:
#                     jitter *= 10
            
#             if L is None: raise np.linalg.LinAlgError("Cholesky failed.")
#             self.cached_inv_R_.append(L)
#             self.cached_alpha_.append(spl.cho_solve((L, True), W_obs[j, :]))

#     def _compute_cross_kernel(self, X1, X2, ls):
#         """Calcule K(X1, X2)"""
#         # Mise à l'échelle
#         X1_s = X1 / ls
#         X2_s = X2 / ls
#         dists = cdist(X1_s, X2_s, metric='sqeuclidean')
        
#         if self.kernel_type == 'rbf':
#             K_mat = np.exp(-0.5 * dists)
#             np.fill_diagonal(K_mat, 1.0)
#             return K_mat
#         elif self.kernel_type == 'matern32':
#             d = np.sqrt(dists)
#             sqrt3 = np.sqrt(3.0)
#             K_mat = (1.0 + sqrt3 * d) * np.exp(-sqrt3 * d)
#             np.fill_diagonal(K_mat, 1.0)
#             return K_mat
    
#     def predict(self, X_new, return_cov=False):
#         """
#         Prédiction optimisée.
        
#         Si return_cov=False: 
#             Retourne (Moyenne, Variance)
#             Moyenne: (N_new, p)
#             Variance: (N_new, p) -> Diagonale seulement, très rapide
            
#         Si return_cov=True:
#             Retourne (Moyenne, Covariance)
#             Covariance: (N_new * p, N_new * p) -> Matrice complète géante
#         """
#         if self.A_hat_ is None:
#             raise Exception("Le modèle n'est pas entraîné (fit).")
            
#         n_new = X_new.shape[0]
        
#         # 1. Prédiction dans l'espace LATENT (W)
#         # On stocke les moyennes et covariances/variances de chaque W_j
#         W_means = np.zeros((self.p, n_new))
        
#         if return_cov:
#             # Liste des matrices de covariance complètes (p matrices de taille N_new x N_new)
#             W_covs = [] 
#         else:
#             # Juste les variances (p vecteurs de taille N_new)
#             W_vars = np.zeros((self.p, n_new))
            
#         for j in range(self.p):
#             ls = self.phis_hat_[j, :]
#             L = self.cached_inv_R_[j]
#             alpha = self.cached_alpha_[j]
            
#             # K_star (N_new, N_train)
#             K_trans = self._compute_cross_kernel(X_new, self.X_train_, ls)
            
#             # Moyenne latente: f* = K_*^T @ alpha
#             f_star = K_trans @ alpha
#             W_means[j, :] = f_star
            
#             if return_cov:
#                 # K_star_star (N_new, N_new) - Pleine
#                 K_star_star = self._compute_cross_kernel(X_new, X_new, ls)
                
#                 # v = L^-1 @ K_star^T
#                 v = spl.solve_triangular(L, K_trans.T, lower=True)
                
#                 # Cov = K_ss - v^T @ v
#                 cov_j = K_star_star - v.T @ v
#                 W_covs.append(cov_j)
#             else:
#                 # OPTIMISATION VARIANCE : K_ss (diag) - sum(v^2)
#                 # On ne calcule que la diagonale de K_star_star (qui vaut 1.0 pour RBF/Matern normalisé)
#                 # k_diag = 1.0 + nugget
#                 k_diag = 1.0 + 1e-6
                
#                 # v = L^-1 @ K_star^T
#                 # On a besoin de v pour le terme quadratique
#                 v = spl.solve_triangular(L, K_trans.T, lower=True)
                
#                 # Var = k_diag - sum(v_i^2) (somme sur les colonnes de v)
#                 var_j = k_diag - np.sum(v**2, axis=0)
#                 # Protection numérique contre variance négative très petite
#                 var_j = np.maximum(var_j, 1e-9)
#                 W_vars[j, :] = var_j
                
#         # 2. Projection dans l'espace OBSERVÉ (V)
#         # Moyenne V = (A @ W)^T
#         V_mean = (self.A_hat_ @ W_means).T
        
#         if not return_cov:
#             # Calcul efficace de la variance marginale
#             # Var(V_i) = sum_k (A_ik^2 * Var(W_k))
#             # On met A au carré élément par élément
#             A_sq = self.A_hat_**2
            
#             # Produit matriciel : (p, p) @ (p, N_new) -> (p, N_new)
#             V_var_T = A_sq @ W_vars
#             return V_mean, V_var_T.T
            
#         else:
#             # Construction de la covariance complète (Lourde !)
#             # Cov(V) = sum_j ( Cov(W_j) Kronecker (a_j * a_j^T) )
#             # Taille finale : (N_new * p, N_new * p)
#             # Pour simplifier l'usage, on retourne souvent une liste ou un tenseur,
#             # mais ici je retourne la matrice complète block par block pour respecter "Toute la covariance"
            
#             # Attention : c'est très coûteux en mémoire.
#             Full_Cov = np.zeros((n_new * self.p, n_new * self.p))
            
#             for j in range(self.p):
#                 # Covariance spatiale du latent j (N_new, N_new)
#                 C_W = W_covs[j]
                
#                 # Structure de corrélation entre tâches (p, p)
#                 a_j = self.A_hat_[:, j].reshape(-1, 1)
#                 C_Task = a_j @ a_j.T
                
#                 # Produit de Kronecker
#                 # Note: l'ordre dépend de comment on veut vectoriser (par point ou par tâche).
#                 # Ici je fais par blocks spatiaux pour correspondre à la logique du papier (Eq 2)
#                 # Mais pour un utilisateur, souvent on veut Cov(V(x), V(x')). 
#                 # Scipy kron standard est le plus simple.
#                 Full_Cov += np.kron(C_Task, C_W) # Check dimensions: (p*N, p*N)
                
#             return V_mean, Full_Cov

#     def sample(self, X_target, n_samples=1):
#         """
#         Échantillonne à partir de la distribution prédictive postérieure.
#         Astuce: Échantillonne W (facile) puis projette V = AW.
        
#         Retourne: (n_samples, n_targets, p_outputs)
#         """
#         if self.A_hat_ is None:
#             raise Exception("Run fit() first.")
            
#         n_new = X_target.shape[0]
        
#         # 1. On a besoin de la covariance complète des LATENTS (spatiale), 
#         # mais indépendante entre latents.
        
#         # Stockage des échantillons latents : (n_samples, p, n_new)
#         W_samples = np.zeros((n_samples, self.p, n_new))
        
#         for j in range(self.p):
#             # Récupération des stats du latent j
#             ls = self.phis_hat_[j, :]
#             L = self.cached_inv_R_[j]
#             alpha = self.cached_alpha_[j]
            
#             # Calcul Covariance (N_new, N_new) et Moyenne
#             K_trans = self._compute_cross_kernel(X_target, self.X_train_, ls)
#             K_ss = self._compute_cross_kernel(X_target, X_target, ls)
            
#             f_star = K_trans @ alpha
            
#             v = spl.solve_triangular(L, K_trans.T, lower=True)
#             cov_star = K_ss - v.T @ v
            
#             # Symmetrisation pour stabilité numérique (mvnrnd est capricieux)
#             cov_star = (cov_star + cov_star.T) / 2
#             cov_star[np.diag_indices_from(cov_star)] += 1e-6
            
#             # Échantillonnage
#             # (n_samples, n_new)
#             try:
#                 # scipy multivariate_normal est parfois lent, on peut utiliser cholesky direct
#                 L_cov = spl.cholesky(cov_star, lower=True)
#                 # mu + L * z
#                 z = np.random.randn(n_new, n_samples)
#                 w_s = f_star[:, None] + L_cov @ z
#                 W_samples[:, j, :] = w_s.T
#             except np.linalg.LinAlgError:
#                 # Fallback robuste
#                 W_samples[:, j, :] = np.random.multivariate_normal(f_star, cov_star, size=n_samples)

#         # 2. Projection V = A * W
#         # W_samples est (n_samples, p, n_new)
#         # A est (p, p)
#         # On veut (n_samples, n_new, p)
        
#         # Produit tensoriel : pour chaque sample, V = A @ W
#         # tensordot ou boucle simple
#         V_samples = np.zeros((n_samples, n_new, self.p))
        
#         for i in range(n_samples):
#             # W_i est (p, n_new)
#             W_i = W_samples[i]
#             # V_i = (A @ W_i).T -> (n_new, p)
#             V_samples[i] = (self.A_hat_ @ W_i).T
            
#         return V_samples

# # # --- TEST ---
# # if __name__ == "__main__":
# #     # Test rapide
# #     model = FastSparseLMC(p_components=2)
# #     # Simulation d'un état "fit" pour tester predict sans lancer l'optim
# #     model.A_hat_ = np.eye(2)
# #     model.phis_hat_ = np.ones((2, 1))
# #     model.X_train_ = np.random.rand(10, 1)
# #     model.Y_train_ = np.random.rand(10, 2)
# #     model.fit(model.X_train_, model.Y_train_) # Juste pour peupler le cache
    
# #     X_test = np.linspace(0, 1, 50).reshape(-1, 1)
    
# #     # 1. Prédiction Variance (Rapide)
# #     mu, var = model.predict(X_test, return_cov=False)
# #     print(f"Prediction (Var only): Mean shape {mu.shape}, Var shape {var.shape}")
    
# #     # 2. Prédiction Covariance (Lente)
# #     mu, cov = model.predict(X_test, return_cov=True)
# #     print(f"Prediction (Full Cov): Mean shape {mu.shape}, Cov shape {cov.shape}")
    
# #     # 3. Échantillonnage
# #     samples = model.sample(X_test, n_samples=3)
# #     print(f"Samples shape: {samples.shape}")

import numpy as np
import scipy.linalg as spl
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import qmc
from joblib import Parallel, delayed

class FastLMCKernel:
    """
    Gère la structure triangulaire inférieure (Cholesky) pour A.
    A = L (Lower Triangular)
    """
    def __init__(self, n_features, p_components, kernel_type='rbf'):
        self.d = n_features
        self.p = p_components
        self.kernel_type = kernel_type
        
        # Nombre d'éléments dans la partie triangulaire inférieure (inclus diagonale)
        self.n_L_params = (self.p * (self.p + 1)) // 2

    def get_parameter_bounds(self, X_min, X_max):
        """
        Définit les bornes :
        - Elements diagonaux de L : [1e-5, 10.0] (Positivité = Inversibilité)
        - Elements hors-diagonale de L : [-10.0, 10.0]
        - Log_Lengthscales : Logarithme des bornes physiques
        """
        bounds_L = []
        
        # On parcourt la matrice L ligne par ligne
        for i in range(self.p):
            for j in range(i + 1):
                if i == j:
                    # Diagonale : Doit être strictement positive
                    bounds_L.append((1e-5, 5.0))
                else:
                    # Hors-diagonale (Triangle inférieur)
                    bounds_L.append((-5.0, 5.0))
        
        # Bornes pour les LOG-lengthscales
        ranges = X_max - X_min
        ranges[ranges == 0] = 1.0 
        
        bounds_log_lengthscales = []
        for _ in range(self.p): 
            for dim in range(self.d):
                lower_phys = 1e-3
                upper_phys = 2.0 * ranges[dim]
                bounds_log_lengthscales.append((np.log(lower_phys), np.log(upper_phys)))
                
        return bounds_L + bounds_log_lengthscales

    def unpack_params(self, theta):
        """
        Reconstruit la matrice L (Triangulaire Inf) et les lengthscales.
        """
        # 1. Reconstruction de L
        L_params = theta[:self.n_L_params]
        L = np.zeros((self.p, self.p))
        
        idx = 0
        for i in range(self.p):
            for j in range(i + 1):
                L[i, j] = L_params[idx]
                idx += 1
                
        # 2. Extraction des lengthscales (Log -> Exp)
        log_ls_params = theta[self.n_L_params:]
        lengthscales = np.exp(log_ls_params).reshape(self.p, self.d)
        
        return L, lengthscales

class FastLMCNll:
    """Calcul de la NLL O(p) avec structure Triangulaire"""
    def __init__(self, kernel, sparsity_lambda=0.0):
        self.kernel = kernel
        self.lam = sparsity_lambda

    def _compute_kernel_matrix(self, X, lengthscales_j):
        # Mise à l'échelle des données
        X_scaled = X / lengthscales_j
        
        # Calcul des distances au carré (vecteur condensé)
        dists_sq_vec = pdist(X_scaled, metric='sqeuclidean')
        
        # Calcul du noyau (vecteur condensé)
        if self.kernel.kernel_type == 'rbf':
            K_vec = np.exp(-0.5 * dists_sq_vec)
        elif self.kernel.kernel_type == 'matern32':
            d_vec = np.sqrt(dists_sq_vec)
            sqrt3 = np.sqrt(3.0)
            K_vec = (1.0 + sqrt3 * d_vec) * np.exp(-sqrt3 * d_vec)
            
        # Transformation en matrice carrée
        K_mat = squareform(K_vec)
        
        # --- CORRECTION CRUCIALE ---
        # squareform met 0 sur la diagonale par défaut. 
        # Or, k(x, x) = 1. On doit remplir la diagonale.
        np.fill_diagonal(K_mat, 1.0)
        
        return K_mat

    def compute(self, theta, X, Y):
        # 1. Déballage
        # Pas besoin de try/except sur unpack, c'est déterministe
        L, lengthscales = self.kernel.unpack_params(theta)
            
        # 2. Inversion de L (Triangulaire)
        # L est triangulaire inférieure. Son déterminant est le produit de la diagonale.
        # Pour le log-det global, on a besoin de 2*n*log|det(L)|
        diag_L = np.diag(L)
        
        # Sécurité : si un élément diag est trop proche de 0 (malgré les bornes)
        # if np.any(diag_L <= 1e-9): 
        #     return 1e15
            
        logdet_L = np.sum(np.log(diag_L))
        
        # W_hat = L^-1 * Y.T
        # solve_triangular est O(p^2) vs O(p^3) pour inv standard
        #try:
        W_hat = spl.solve_triangular(L, Y.T, lower=True)
        # except np.linalg.LinAlgError:
        #     return 1e15

        # 3. Somme sur les processus latents
        log_det_Rs = 0
        quad_form_sum = 0
        n = X.shape[0]
        p = Y.shape[1]
       # print('p : ', theta)
        for j in range(p):
            #print('j : ', j)
            #print("lengthscales for process", j, ":", lengthscales[j, :])
            R_j = self._compute_kernel_matrix(X, lengthscales[j, :])
            R_j[np.diag_indices_from(R_j)] += 1e-6 
            #print("R_j :", R_j)
            
            # try:
            chol_R = spl.cholesky(R_j, lower=True)
            log_det_Rs += 2 * np.sum(np.log(np.diag(chol_R)))
            w_j = W_hat[j, :]
            alpha = spl.cho_solve((chol_R, True), w_j)
            quad_form_sum += w_j @ alpha
            # except np.linalg.LinAlgError:
            #     return 1e15

        const = n * p * np.log(2 * np.pi)
        
        # log(det(Sigma)) = 2*n*log(det(L)) + sum(log(det(R_j)))
        total_log_det = 2 * n * logdet_L + log_det_Rs
        
        nll = 0.5 * (const + total_log_det + quad_form_sum)
        
        # Sparsité sur L (Lasso)
        if self.lam > 0:
            nll += self.lam * np.sum(np.abs(L))
            
        return nll

class FastSparseLMC:
    def __init__(self, p_components, kernel_type='rbf', sparsity_lambda=0.0, n_restarts=5, use_init_heuristic=True, n_jobs=1,seed = 42):
        self.p = p_components
        self.kernel_type = kernel_type
        self.lam = sparsity_lambda
        self.n_restarts = n_restarts
        self.use_init_heuristic = use_init_heuristic # Renommé pour généralité (c'était use_pca)
        self.n_jobs = n_jobs
        self.seed = seed
        
        self.kernel = None 
        self.best_params_ = None
        self.cached_inv_R_ = []
        self.cached_alpha_ = []
        self.L_hat_ = None # On stocke L maintenant
        self.phis_hat_ = None

    def _initialize_hyperparams(self, n_starts, bounds):
        n_params = len(bounds)
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        sampler = qmc.LatinHypercube(d=n_params, optimization="random-cd", rng=self.seed)
        sample = sampler.random(n=n_starts)
        initial_thetas = qmc.scale(sample, lower_bounds, upper_bounds)
        
        # --- INITIALISATION INTELLIGENTE (CHOLESKY EMPIRIQUE) ---
        if self.use_init_heuristic and self.Y_train_ is not None:
            try:
                # 1. Calcul de la covariance empirique des sorties Y
                # (p, p)
                cov_emp = np.cov(self.Y_train_.T)
                
                # Jitter pour garantir que la cov empirique est définie positive
                # (nécessaire si n < p ou données très corrélées)
                cov_emp += np.eye(self.p) * 1e-4
                
                # 2. Décomposition de Cholesky : Cov = L_init * L_init.T
                # C'est exactement la structure que nous cherchons !
                L_init = np.linalg.cholesky(cov_emp)
                
                # Clipping pour rester dans les bornes [-10, 10]
                L_init = np.clip(L_init, -10.0, 10.0)
                # Diagonale positive
                np.fill_diagonal(L_init, np.maximum(np.diag(L_init), 1e-4))
                
                # 3. Packing dans le vecteur theta
                L_flat = []
                for i in range(self.p):
                    for j in range(i + 1):
                        L_flat.append(L_init[i, j])
                L_flat = np.array(L_flat)
                
                # Appliquer à tous les starts (on garde la diversité des lengthscales)
                # Les params de L sont au début du vecteur
                n_L = len(L_flat)
                for i in range(n_starts):
                    initial_thetas[i, :n_L] = L_flat
                    
            except np.linalg.LinAlgError:
                print("Warning: Cholesky init failed, using LHS random.")

        return initial_thetas, list(zip(lower_bounds, upper_bounds))

    def _optimize_hyperparams(self, initial_theta, bounds):
        #try:
        res = minimize(
            self.nll_engine.compute,
            initial_theta,
            args=(self.X_train_, self.Y_train_),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200} #, 'ftol': 1e-9}
        )
        return res
        # except Exception:
        #     return None

    def fit(self, X, Y):
        # Recommandation à l'utilisateur
        if np.max(np.abs(Y)) > 100:
            print("ATTENTION: Vos données Y ont une grande amplitude.")
            print("Il est fortement recommandé de normaliser Y (StandardScaler) avant le fit")
            print("car les bornes de la matrice L sont fixées à [-10, 10].")

        self.X_train_ = X
        self.Y_train_ = Y
        n_features = X.shape[1]
        
        self.kernel = FastLMCKernel(n_features, self.p, self.kernel_type)
        self.nll_engine = FastLMCNll(self.kernel, self.lam)
        
        # Bornes
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        bounds_def = self.kernel.get_parameter_bounds(X_min, X_max)
        
        
        # Init
        starts, bounds = self._initialize_hyperparams(self.n_restarts, bounds_def)
        
        # Optim
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._optimize_hyperparams)(theta, bounds) for theta in starts
        )
        
        valid_results = [r for r in results if r is not None and r.success]
        if not valid_results: valid_results = [r for r in results if r is not None]
        if not valid_results: raise RuntimeError("Optim failed.")
            
        best_res = min(valid_results, key=lambda x: x.fun)
        self.best_params_ = best_res.x
        self.final_nll_ = best_res.fun
        
        # --- MISE EN CACHE ---
        self.L_hat_, self.phis_hat_ = self.kernel.unpack_params(self.best_params_)
        
        # W_obs = L^-1 * Y.T
        W_obs = spl.solve_triangular(self.L_hat_, self.Y_train_.T, lower=True)
        
        self.cached_inv_R_ = []
        self.cached_alpha_ = []
        
        for j in range(self.p):
            ls = self.phis_hat_[j, :]
            X_scaled = self.X_train_ / ls
            dists = squareform(pdist(X_scaled, metric='sqeuclidean'))
            
            if self.kernel_type == 'rbf': K_base = np.exp(-0.5 * dists)
            elif self.kernel_type == 'matern32':
                d = np.sqrt(dists); sqrt3 = np.sqrt(3.0)
                K_base = (1.0 + sqrt3 * d) * np.exp(-sqrt3 * d)
            
            np.fill_diagonal(K_base, 1.0)
            # Robust Cholesky Jitter
            jitter = 1e-6
            L_chol = None
            for _ in range(5):
                try:
                    K_mat = K_base.copy()
                    K_mat[np.diag_indices_from(K_mat)] += jitter
                    L_chol = spl.cholesky(K_mat, lower=True)
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10
            
            if L_chol is None: raise np.linalg.LinAlgError("Cholesky failed.")
            self.cached_inv_R_.append(L_chol)
            self.cached_alpha_.append(spl.cho_solve((L_chol, True), W_obs[j, :]))

    def predict(self, X_new, return_cov=False):
        return self._predict_impl(X_new, return_cov)

    def _predict_impl(self, X_new, return_cov):
        if self.L_hat_ is None: raise Exception("Not fitted")
        n_new = X_new.shape[0]
        W_means = np.zeros((self.p, n_new))
        
        if return_cov: W_covs = []
        else: W_vars = np.zeros((self.p, n_new))
            
        for j in range(self.p):
            ls = self.phis_hat_[j, :]
            L_chol = self.cached_inv_R_[j]
            alpha = self.cached_alpha_[j]
            
            X1_s = X_new / ls; X2_s = self.X_train_ / ls
            dists_c = cdist(X1_s, X2_s, metric='sqeuclidean')
            
            if self.kernel_type == 'rbf': K_trans = np.exp(-0.5 * dists_c)
            else:
                d = np.sqrt(dists_c); sqrt3 = np.sqrt(3.0)
                K_trans = (1.0 + sqrt3 * d) * np.exp(-sqrt3 * d)
            
            f_star = K_trans @ alpha
            W_means[j, :] = f_star
            
            v = spl.solve_triangular(L_chol, K_trans.T, lower=True)
            
            if return_cov:
                X_s = X_new / ls
                d_ss = pdist(X_s, metric='sqeuclidean')
                if self.kernel_type == 'rbf': K_ss = squareform(np.exp(-0.5 * d_ss))
                else:
                    d = np.sqrt(d_ss); K_ss = squareform((1.0 + sqrt3*d)*np.exp(-sqrt3*d))
                K_ss[np.diag_indices_from(K_ss)] += 1.0
                
                cov_j = K_ss - v.T @ v
                W_covs.append(cov_j)
            else:
                k_diag = 1.0 + 1e-6
                var_j = np.maximum(k_diag - np.sum(v**2, axis=0), 1e-9)
                W_vars[j, :] = var_j
        
        # Projection : V = L * W
        V_mean = (self.L_hat_ @ W_means).T
        
        if not return_cov:
            # Var(V) = (L^2) * Var(W)
            L_sq = self.L_hat_**2
            V_var_T = L_sq @ W_vars
            return V_mean, V_var_T.T
        else:
            Full_Cov = np.zeros((n_new * self.p, n_new * self.p))
            for j in range(self.p):
                C_W = W_covs[j]
                # Partie spatiale du latent j
                # Produit Kronecker avec la structure de L pour ce latent
                l_j = self.L_hat_[:, j].reshape(-1, 1)
                Full_Cov += np.kron(l_j @ l_j.T, C_W)
            return V_mean, Full_Cov