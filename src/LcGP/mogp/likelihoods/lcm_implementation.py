import numpy as np
from scipy.linalg import cholesky, cho_solve, inv, det
import matplotlib.pyplot as plt
from tqdm import tqdm

class NonSeparableLMC:
    """
    Implémentation du modèle de corégionalisation linéaire (LMC) non séparable.
    
    Le modèle s'écrit Z = Aw où:
    - Z est un processus composé de p processus
    - A est une matrice de rang complet (p×p)
    - w est un vecteur de p processus gaussiens indépendants avec des noyaux R_j
    """
    
    def __init__(self, p, kernel_functions, A=None, nugget=1e-8):
        """
        Initialisation du modèle LMC non séparable.
        
        Args:
            p: Nombre de processus de sortie
            kernel_functions: Liste de p fonctions de noyau pour les processus latents
            A: Matrice de transformation (p×p), si None une matrice aléatoire est générée
            nugget: Terme de régularisation numérique pour stabiliser les calculs
        """
        self.p = p
        self.kernel_functions = kernel_functions
        self.nugget = nugget
        
        # Initialiser A si non fournie
        if A is None:
            self.A = np.random.randn(p, p)
        else:
            self.A = A
            
        # Vérifier que A est de rang complet
        if np.linalg.matrix_rank(self.A) != p:
            raise ValueError("La matrice A doit être de rang complet")
    
    def compute_kernel_matrices(self, X):
        """
        Calcule les matrices de noyau R_j pour chaque processus latent.
        
        Args:
            X: Points d'entrée de forme (n, d) où n est le nombre de points et d la dimension
            
        Returns:
            Liste de p matrices de noyau (n×n)
        """
        n = X.shape[0]
        R_matrices = []
        
        for j in range(self.p):
            K = np.zeros((n, n))
            for i in range(n):
                for k in range(n):
                    K[i, k] = self.kernel_functions[j](X[i], X[k])
            
            # Ajouter un terme de nugget pour la stabilité numérique
            np.fill_diagonal(K, K.diagonal() + self.nugget)
            R_matrices.append(K)
            
        return R_matrices
    
    def log_likelihood_efficient(self, X, Y):
        """
        Calcule la log-vraisemblance du modèle LMC selon la proposition 1.
        
        Args:
            X: Points d'entrée de forme (n, d)
            Y: Observations de forme (p, n) où chaque ligne correspond à un processus
            
        Returns:
            La log-vraisemblance
        """
        n = X.shape[0]
        
        # Calculer les matrices de noyau R_j
        R_matrices = self.compute_kernel_matrices(X)
        
        # Calculer l'inverse de A
        A_inv = inv(self.A)
       # print("A : ", A)
        
        # Calculer V = Y
        V = Y
        
        # Premier terme: -1/2 * sum_{j=1}^p [a_j^{-1} V R_j^{-1} V^T a_j^{-T}]
        term1 = 0
        for j in range(self.p):
            # On utilise la décomposition de Cholesky pour résoudre les systèmes linéaires plus efficacement
            L_j = cholesky(R_matrices[j], lower=True)
            
            # a_j^{-1} est la j-ème ligne de A^{-1}
            a_j_inv = A_inv[j]
            
            # Calculer R_j^{-1} V^T
            R_j_inv_VT = cho_solve((L_j, True), V.T)
            
            # Calculer V R_j^{-1} V^T
            V_R_j_inv_VT = V.dot(R_j_inv_VT)
            
            # Calculer a_j^{-1} V R_j^{-1} V^T a_j^{-T}
            term = np.dot(a_j_inv, np.dot(V_R_j_inv_VT, a_j_inv.T))
            #print("term1 : ", term)
            term1 += term
            
        term1 = -0.5 * term1
        
        # Deuxième terme: -n*p/2 * log(2π)
        term2 = -n * self.p / 2 * np.log(2 * np.pi)
        
        # Troisième terme: -n * log|A|
        term3 = -n * np.log(np.abs(det(self.A)))
        #print("term 3 : ", term3)
        # Quatrième terme: -1/2 * sum_{j=1}^p log|R_j|
        term4 = 0
        for j in range(self.p):
            # Utiliser le produit des éléments diagonaux de L_j pour calculer det(R_j)
            L_j = cholesky(R_matrices[j], lower=True)
            log_det_R_j = 2 * np.sum(np.log(np.diag(L_j)))
            term4 -= 0.5 * log_det_R_j
            
        # Somme de tous les termes
        log_likelihood = term1 + term2 + term3 + term4
        
        return log_likelihood
    
    def log_likelihood_naive(self, X, Y):
        """
        Calcule la log-vraisemblance du modèle LMC de manière naïve en construisant 
        explicitement la matrice de covariance complète.
        
        Args:
            X: Points d'entrée de forme (n, d)
            Y: Observations de forme (p, n) où chaque ligne correspond à un processus
            
        Returns:
            La log-vraisemblance
        """
        n = X.shape[0]
        
        # Calculer les matrices de noyau R_j
        R_matrices = self.compute_kernel_matrices(X)
        
        # Construire la matrice de covariance complète Sigma = sum_{j=1}^p R_j ⊗ a_j a_j^T
        Sigma = np.zeros((n * self.p, n * self.p))
        for j in range(self.p):
            a_j = self.A[:, j].reshape(-1, 1)
            a_j_a_jT = np.dot(a_j, a_j.T)
            
            # Calcul direct du produit de Kronecker
            for i1 in range(self.p):
                for i2 in range(self.p):
                    Sigma[i1*n:(i1+1)*n, i2*n:(i2+1)*n] += a_j_a_jT[i1, i2] * R_matrices[j]
        
        # Ajouter un nugget sur la diagonale pour la stabilité
        np.fill_diagonal(Sigma, Sigma.diagonal() + self.nugget)
        
        # Vectoriser Y
        y_vec = Y.T.flatten()
        
        # Calculer la log-vraisemblance multivariate normale
        try:
            L = cholesky(Sigma, lower=True)
            alpha = cho_solve((L, True), y_vec)
            
            # Formule de la log-vraisemblance
            log_det = 2 * np.sum(np.log(np.diag(L)))
            print("log_det : ", log_det)
            log_likelihood = -0.5 * (np.dot(y_vec, alpha) + log_det + n * self.p * np.log(2 * np.pi))
        except np.linalg.LinAlgError:
            # En cas d'échec de la décomposition de Cholesky
            print("Avertissement: La décomposition de Cholesky a échoué, utilisation de la méthode d'inversion directe")
            inv_Sigma = inv(Sigma)
            log_likelihood = -0.5 * (
                np.dot(y_vec, np.dot(inv_Sigma, y_vec)) + 
                np.log(det(Sigma)) + 
                n * self.p * np.log(2 * np.pi)
            )
            
        return log_likelihood
    
    def generate_samples(self, X):
        """
        Génère des échantillons du processus Z en utilisant l'équation Z = sum_j a_j z_j B_j^T.
        
        Args:
            X: Points d'entrée de forme (n, d)
            
        Returns:
            Échantillons du processus Z de forme (p, n)
        """
        n = X.shape[0]
        
        # Calculer les matrices de noyau R_j et leurs facteurs de Cholesky
        R_matrices = self.compute_kernel_matrices(X)
        B_matrices = [cholesky(R, lower=True) for R in R_matrices]
        
        # Initialiser la matrice V (résultat)
        V = np.zeros((self.p, n))
        
        # Pour chaque processus latent
        for j in range(self.p):
            # Générer des échantillons gaussiens standard
            z_j = np.random.normal(0, 1, n)
            
            # Calculer B_j^T * z_j
            B_j_T_z_j = B_matrices[j].T @ z_j
            
            # Ajouter la contribution à V
            a_j = self.A[:, j].reshape(-1, 1)
            V += np.dot(a_j, B_j_T_z_j.reshape(1, -1))
            
        return V

# Définition de quelques noyaux
def rbf_kernel(length_scale=1.0):
    def kernel(x, y):
        return np.exp(-0.5 * np.sum((x - y) ** 2) / (length_scale ** 2))
    return kernel

def matern32_kernel(length_scale=1.0):
    def kernel(x, y):
        d = np.sqrt(np.sum((x - y) ** 2))
        d_scaled = np.sqrt(3) * d / length_scale
        return (1 + d_scaled) * np.exp(-d_scaled)
    return kernel

def matern52_kernel(length_scale=1.0):
    def kernel(x, y):
        d = np.sqrt(np.sum((x - y) ** 2))
        d_scaled = np.sqrt(5) * d / length_scale
        return (1 + d_scaled + d_scaled**2/3) * np.exp(-d_scaled)
    return kernel


# Test de l'implémentation
if __name__ == "__main__":
    # Paramètres
    p = 3  # Nombre de processus
    n = 10  # Nombre de points de données
    d = 1   # Dimension des entrées
    np.random.seed(42)
    # Générer des points d'entrée aléatoires
    X = np.random.rand(n, d)
    
    # Définir les noyaux pour chaque processus latent
    kernel_functions = [
        rbf_kernel(length_scale=1.5),
        matern32_kernel(length_scale=1.3),
        matern52_kernel(length_scale=1.7)
    ]
    
    # Créer une matrice A aléatoire de rang complet
    
    A = np.random.randn(p, p)
    
    # Initialiser le modèle LMC
    lmc = NonSeparableLMC(p, kernel_functions, A, nugget=1e-6)
    
    # Générer des échantillons
    Y = lmc.generate_samples(X) #+10*np.random.randn(p,n)
    
    # Calculer la log-vraisemblance avec les deux méthodes
    ll_efficient = lmc.log_likelihood_efficient(X, Y)
    ll_naive = lmc.log_likelihood_naive(X, Y)
    
    print(f"Log-vraisemblance (méthode efficace): {ll_efficient}")
    print(f"Log-vraisemblance (méthode naïve): {ll_naive}")
    print(f"Différence: {abs(ll_efficient - ll_naive)}")
    
    # Comparaison des performances
    n_values = [10, 20, 30, 40, 50, 60, 70, 80]
    time_efficient = []
    time_naive = []
    
    import time
    
    for n_test in n_values:
        X_test = np.random.rand(n_test, d)
        lmc = NonSeparableLMC(p, kernel_functions, A, nugget=1e-6)
        Y_test = lmc.generate_samples(X_test)
        
        # Temps pour la méthode efficace
        start = time.time()
        _ = lmc.log_likelihood_efficient(X_test, Y_test)
        end = time.time()
        time_efficient.append(end - start)
        
        # Temps pour la méthode naïve
        start = time.time()
        _ = lmc.log_likelihood_naive(X_test, Y_test)
        end = time.time()
        time_naive.append(end - start)
    
    # Visualisation des résultats
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, time_efficient, 'o-', label='Méthode efficace')
    plt.plot(n_values, time_naive, 'o-', label='Méthode naïve')
    plt.xlabel('Nombre de points (n)')
    plt.ylabel('Temps d\'exécution (s)')
    plt.title('Comparaison des performances des méthodes de calcul de log-vraisemblance')
    plt.legend()
    plt.grid(True)
    plt.savefig('lmc_performance_comparison.png')
    plt.show()
    
    # Plot de la variance du modèle pour illustration
    if d == 1:
        X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
        samples = np.array([lmc.generate_samples(X_plot) for _ in range(10)])
        
        plt.figure(figsize=(15, 5))
        for i in range(p):
            plt.subplot(1, p, i+1)
            for s in range(samples.shape[0]):
                plt.plot(X_plot, samples[s, i, :], 'b-', alpha=0.3)
            plt.title(f'Processus {i+1}')
            plt.grid(True)
        plt.tight_layout()
        plt.savefig('lmc_samples.png')
        plt.show()