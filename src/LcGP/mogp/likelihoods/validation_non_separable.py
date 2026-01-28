import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.stats import norm

# Importer la classe NonSeparableLMC du fichier précédent
from lcm_implementation import NonSeparableLMC, rbf_kernel, matern32_kernel, matern52_kernel

def validate_likelihood_accuracy():
    """
    Valide l'exactitude de l'implémentation de la vraisemblance efficace
    en la comparant à la méthode naïve pour différentes tailles de données.
    """
    p = 2  # Nombre de processus réduit pour la validation
    d = 1   # Dimension des entrées
    
    # Définir les noyaux pour chaque processus latent
    kernel_functions = [
        rbf_kernel(length_scale=0.5),
        matern32_kernel(length_scale=0.3)
    ]
    
    # Créer une matrice A fixe pour la validation
    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    
    n_values = [5, 10, 15, 20, 25]
    diff_values = []
    rel_diff_values = []
    
    for n in n_values:
        # Générer des points d'entrée aléatoires
        X = np.linspace(0, 1, n).reshape(-1, 1)
        
        # Initialiser le modèle LMC
        lmc = NonSeparableLMC(p, kernel_functions, A, nugget=1e-6)
        
        # Générer des échantillons
        Y = lmc.generate_samples(X)
        
        # Calculer la log-vraisemblance avec les deux méthodes
        ll_efficient = lmc.log_likelihood_efficient(X, Y)
        ll_naive = lmc.log_likelihood_naive(X, Y)
        
        diff = abs(ll_efficient - ll_naive)
        rel_diff = diff / abs(ll_naive) if ll_naive != 0 else 0
        
        diff_values.append(diff)
        rel_diff_values.append(rel_diff)
        
        print(f"n={n}: Diff={diff:.6f}, Rel_diff={rel_diff:.6f}")
    
    # Visualisation des différences
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(n_values, diff_values, 'o-')
    plt.xlabel('Nombre de points (n)')
    plt.ylabel('Différence absolue')
    plt.title('Différence absolue entre les méthodes')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(n_values, rel_diff_values, 'o-')
    plt.xlabel('Nombre de points (n)')
    plt.ylabel('Différence relative')
    plt.title('Différence relative entre les méthodes')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('lmc_validation_accuracy.png')
    
    return diff_values, rel_diff_values

def benchmark_performance(max_n=100):
    """
    Compare les performances des deux méthodes pour des tailles croissantes de données.
    """
    p = 3  # Nombre de processus
    d = 1   # Dimension des entrées
    
    # Définir les noyaux pour chaque processus latent
    kernel_functions = [
        rbf_kernel(length_scale=0.5),
        matern32_kernel(length_scale=0.3),
        matern52_kernel(length_scale=0.7)
    ]
    
    # Créer une matrice A aléatoire de rang complet
    A = np.random.randn(p, p)
    
    # Définir les tailles à tester
    n_values = np.linspace(10, max_n, 10, dtype=int)
    time_efficient = []
    time_naive = []
    
    for n_test in tqdm(n_values, desc="Benchmark des performances"):
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
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(n_values, time_efficient, 'o-', label='Méthode efficace')
    plt.plot(n_values, time_naive, 'o-', label='Méthode naïve')
    plt.xlabel('Nombre de points (n)')
    plt.ylabel('Temps d\'exécution (s)')
    plt.title('Comparaison des temps d\'exécution')
    plt.legend()
    plt.grid(True)
    
    # Speedup
    plt.subplot(1, 2, 2)
    speedup = np.array(time_naive) / np.array(time_efficient)
    plt.plot(n_values, speedup, 'o-')
    plt.xlabel('Nombre de points (n)')
    plt.ylabel('Accélération (naive/efficace)')
    plt.title('Accélération de la méthode efficace')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('lmc_benchmark_performance.png')
    
    return n_values, time_efficient, time_naive, speedup

def validate_with_optim(n_runs=20):
    """
    Valide l'implémentation en optimisant les paramètres du modèle
    et en vérifiant si on peut retrouver les paramètres originaux.
    """
    from scipy.optimize import minimize
    
    p = 2  # Nombre de processus réduit pour la validation
    n = 40  # Nombre de points
    d = 1   # Dimension des entrées
    
    successes = 0
    relative_errors = []
    
    for run in tqdm(range(n_runs), desc="Validation par optimisation"):
        # Générer des points d'entrée aléatoires
        X = np.linspace(0, 1, n).reshape(-1, 1)
        
        # Paramètres réels
        true_length_scales = [0.5, 0.3]
        
        # Définir les noyaux avec les vrais paramètres
        def create_kernel_functions(params):
            return [
                rbf_kernel(length_scale=params[0]),
                matern32_kernel(length_scale=params[1])
            ]
        
        # Matrice A (fixée pour simplifier)
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        # Initialiser le modèle LMC avec les vrais paramètres
        true_kernels = create_kernel_functions(true_length_scales)
        true_lmc = NonSeparableLMC(p, true_kernels, A, nugget=1e-6)
        
        # Générer des échantillons
        Y = true_lmc.generate_samples(X)
        
        # Fonction objectif pour l'optimisation
        def objective(params):
            if np.any(params <= 0.01) or np.any(params > 2.0):  # Limites des paramètres
                return 1e6
            
            kernels = create_kernel_functions(params)
            model = NonSeparableLMC(p, kernels, A, nugget=1e-6)
            return -model.log_likelihood_efficient(X, Y)
        
        # Optimisation
        initial_guess = [0.1, 0.1]  # Valeurs initiales différentes des vraies valeurs
        result = minimize(objective, initial_guess, method='L-BFGS-B', 
                         bounds=[(0.01, 2.0), (0.01, 2.0)])
        
        # Vérifier si l'optimisation a réussi
        est_length_scales = result.x
        
        # Calculer l'erreur relative moyenne
        rel_error = np.mean(np.abs((est_length_scales - true_length_scales) / true_length_scales))
        relative_errors.append(rel_error)
        
        # Considérer comme un succès si l'erreur relative moyenne est inférieure à 20%
        if rel_error < 0.2:
            successes += 1
            
        print(f"Run {run+1}: Vrai={true_length_scales}, Estimé={est_length_scales}, "
              f"Erreur relative={rel_error:.4f}")
    
    # Résultats
    success_rate = successes / n_runs
    avg_rel_error = np.mean(relative_errors)
    
    print(f"\nRésultats après {n_runs} exécutions:")
    print(f"Taux de succès: {success_rate:.2f} ({successes}/{n_runs})")
    print(f"Erreur relative moyenne: {avg_rel_error:.4f}")
    
    # Visualisation des erreurs relatives
    plt.figure(figsize=(10, 6))
    plt.hist(relative_errors, bins=10)
    plt.axvline(avg_rel_error, color='r', linestyle='--', label=f'Moyenne: {avg_rel_error:.4f}')
    plt.xlabel('Erreur relative')
    plt.ylabel('Fréquence')
    plt.title('Distribution des erreurs relatives')
    plt.legend()
    plt.grid(True)
    plt.savefig('lmc_optimization_validation.png')
    
    return success_rate, avg_rel_error, relative_errors

def visualize_samples():
    """
    Visualise des échantillons du modèle LMC pour différentes configurations.
    """
    p = 3  # Nombre de processus
    n = 100  # Nombre de points
    d = 1   # Dimension des entrées
    
    # Générer des points d'entrée réguliers
    X = np.linspace(0, 1, n).reshape(-1, 1)
    
    # Différentes configurations de noyaux
    kernel_configs = [
        [rbf_kernel(length_scale=0.1), rbf_kernel(length_scale=0.3), rbf_kernel(length_scale=0.5)],
        [matern32_kernel(length_scale=0.2), matern32_kernel(length_scale=0.4), matern32_kernel(length_scale=0.6)],
        [rbf_kernel(length_scale=0.3), matern32_kernel(length_scale=0.3), matern52_kernel(length_scale=0.3)]
    ]
    
    config_names = ["RBF avec différentes échelles", "Matern32 avec différentes échelles", "Différents types de noyaux"]
    
    # Différentes matrices A
    A_configs = [
        np.eye(p),  # Matrice identité (pas de corrélation entre les sorties)
        np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]),  # Corrélation modérée
        np.array([[1.0, 0.8, 0.7], [0.8, 1.0, 0.9], [0.7, 0.9, 1.0]])   # Forte corrélation
    ]
    
    A_names = ["Sans corrélation", "Corrélation modérée", "Forte corrélation"]
    
    # Générer et visualiser les échantillons
    n_samples = 5  # Nombre d'échantillons par configuration
    
    for k_idx, kernels in enumerate(kernel_configs):
        for a_idx, A in enumerate(A_configs):
            plt.figure(figsize=(15, 10))
            
            # Initialiser le modèle
            lmc = NonSeparableLMC(p, kernels, A, nugget=1e-6)
            
            # Générer plusieurs échantillons
            samples = np.array([lmc.generate_samples(X) for _ in range(n_samples)])
            
            # Tracer les échantillons
            for i in range(p):
                plt.subplot(p, 1, i+1)
                for s in range(samples.shape[0]):
                    plt.plot(X, samples[s, i, :], alpha=0.7)
                plt.title(f'Processus {i+1}')
                plt.grid(True)
                
                if i == 0:
                    plt.suptitle(f"Noyaux: {config_names[k_idx]}, Matrice A: {A_names[a_idx]}")
            
            plt.tight_layout()
            plt.savefig(f'lmc_samples_k{k_idx+1}_a{a_idx+1}.png')
            plt.close()
            
    return "Visualisation des échantillons terminée"

if __name__ == "__main__":
    # Exécuter les validations et benchmarks
    print("1. Validation de l'exactitude de la vraisemblance...")
    diff_values, rel_diff_values = validate_likelihood_accuracy()
    
    print("\n2. Benchmark des performances...")
    n_values, time_efficient, time_naive, speedup = benchmark_performance(max_n=100)
    
    print("\n3. Validation par optimisation des paramètres...")
    success_rate, avg_rel_error, relative_errors = validate_with_optim(n_runs=10)
    
    print("\n4. Visualisation des échantillons...")
    status = visualize_samples()
    
    print("\nToutes les validations sont terminées.")