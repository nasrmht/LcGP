import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import cholesky, solve_triangular, LinAlgError
import warnings
warnings.filterwarnings('ignore')

def make_positive_definite(matrix, jitter=1e-6):
    """Rend une matrice définie positive en ajoutant du jitter sur la diagonale."""
    return matrix + jitter * np.eye(matrix.shape[0])

def generate_test_matrices(N, P, seed=42):
    """Génère des matrices de test pour le benchmark."""
    np.random.seed(seed)
    
    # Matrice globale NP x NP
    A_global = np.random.randn(N*P, N*P)
    K_global = A_global @ A_global.T
    K_global = make_positive_definite(K_global)
    
    # P matrices N x N (blocs diagonaux)
    K_blocks = []
    for p in range(P):
        A_block = np.random.randn(N, N)
        K_block = A_block @ A_block.T
        K_block = make_positive_definite(K_block)
        K_blocks.append(K_block)
    
    # Vecteurs Y correspondants
    Y_global = np.random.randn(N*P)
    Y_blocks = [np.random.randn(N) for _ in range(P)]
    
    return K_global, K_blocks, Y_global, Y_blocks

# ============= FONCTIONS D'INVERSION =============

def invert_global_matrix(K):
    """Inversion directe de la matrice globale NP x NP."""
    try:
        return np.linalg.inv(K)
    except LinAlgError:
        return None

def invert_block_matrices(K_blocks):
    """Inversion de P matrices N x N séparément."""
    try:
        return [np.linalg.inv(K) for K in K_blocks]
    except LinAlgError:
        return None

def cholesky_invert_global(K):
    """Inversion via décomposition de Cholesky (matrice globale)."""
    try:
        L = cholesky(K, lower=True)
        # Inversion de la matrice triangulaire
        L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True)
        return L_inv.T @ L_inv
    except LinAlgError:
        return None

def cholesky_invert_blocks(K_blocks):
    """Inversion via décomposition de Cholesky (matrices par blocs)."""
    try:
        inverted_blocks = []
        for K in K_blocks:
            L = cholesky(K, lower=True)
            L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True)
            inverted_blocks.append(L_inv.T @ L_inv)
        return inverted_blocks
    except LinAlgError:
        return None

# ============= FONCTIONS DE CALCUL QUADRATIQUE =============

def quadratic_form_global(Y, K):
    """Calcul de Y^T @ K^-1 @ Y (approche globale)."""
    try:
        K_inv = np.linalg.inv(K)
        return Y.T @ K_inv @ Y
    except LinAlgError:
        return None

def quadratic_form_blocks(Y_blocks, K_blocks):
    """Calcul de somme des y^T @ k^-1 @ y (approche par blocs)."""
    try:
        total = 0.0
        for y, K in zip(Y_blocks, K_blocks):
            K_inv = np.linalg.inv(K)
            total += y.T @ K_inv @ y
        return total
    except LinAlgError:
        return None

def quadratic_form_global_cholesky(Y, K):
    """Calcul de Y^T @ K^-1 @ Y via Cholesky (approche globale)."""
    try:
        L = cholesky(K, lower=True)
        # Résolution de L @ z = Y
        z = solve_triangular(L, Y, lower=True)
        return np.sum(z**2)
    except LinAlgError:
        return None

def quadratic_form_blocks_cholesky(Y_blocks, K_blocks):
    """Calcul de somme des y^T @ k^-1 @ y via Cholesky (approche par blocs)."""
    try:
        total = 0.0
        for y, K in zip(Y_blocks, K_blocks):
            L = cholesky(K, lower=True)
            z = solve_triangular(L, y, lower=True)
            total += np.sum(z**2)
        return total
    except LinAlgError:
        return None

# ============= BENCHMARK =============

def benchmark_operation(func, *args, n_repeats=10):
    """Benchmark d'une opération avec répétitions."""
    start_time = time.perf_counter()
    success_count = 0
    for _ in range(n_repeats):
        result = func(*args)
        if result is not None:
            success_count += 1
    end_time = time.perf_counter()
    
    if success_count > 0:
        return end_time - start_time
    else:
        return np.nan

def run_benchmark(N=50, P_values=range(1, 11), n_repeats=5):
    """Exécute le benchmark complet."""
    results = {
        'P_values': P_values,
        'inversion_global': [],
        'inversion_blocks': [],
        'inversion_chol_global': [],
        'inversion_chol_blocks': [],
        'quad_global': [],
        'quad_blocks': [],
        'quad_chol_global': [],
        'quad_chol_blocks': []
    }
    
    print(f"Benchmark pour N={N}, P variant de {min(P_values)} à {max(P_values)}")
    print("=" * 60)
    
    for P in P_values:
        print(f"Testing P={P}...")
        
        # Génération des données de test
        K_global, K_blocks, Y_global, Y_blocks = generate_test_matrices(N, P)
        
        # Benchmark des inversions
        total_time = benchmark_operation(invert_global_matrix, K_global, n_repeats=n_repeats)
        results['inversion_global'].append(total_time)
        
        total_time = benchmark_operation(invert_block_matrices, K_blocks, n_repeats=n_repeats)
        results['inversion_blocks'].append(total_time)
        
        total_time = benchmark_operation(cholesky_invert_global, K_global, n_repeats=n_repeats)
        results['inversion_chol_global'].append(total_time)
        
        total_time = benchmark_operation(cholesky_invert_blocks, K_blocks, n_repeats=n_repeats)
        results['inversion_chol_blocks'].append(total_time)
        
        # Benchmark des formes quadratiques
        total_time = benchmark_operation(quadratic_form_global, Y_global, K_global, n_repeats=n_repeats)
        results['quad_global'].append(total_time)
        
        total_time = benchmark_operation(quadratic_form_blocks, Y_blocks, K_blocks, n_repeats=n_repeats)
        results['quad_blocks'].append(total_time)
        
        total_time = benchmark_operation(quadratic_form_global_cholesky, Y_global, K_global, n_repeats=n_repeats)
        results['quad_chol_global'].append(total_time)
        
        total_time = benchmark_operation(quadratic_form_blocks_cholesky, Y_blocks, K_blocks, n_repeats=n_repeats)
        results['quad_chol_blocks'].append(total_time)
    
    return results

def plot_results(results):
    """Crée les graphiques des résultats."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    P_values = results['P_values']
    
    # Graphique 1: Inversion directe
    ax1.plot(P_values, results['inversion_global'], 
             label='Inversion globale (NP×NP)', marker='o', linewidth=2)
    ax1.plot(P_values, results['inversion_blocks'], 
             label='Inversion par blocs (P × N×N)', marker='s', linewidth=2)
    ax1.set_xlabel('Nombre de blocs P')
    ax1.set_ylabel('Temps total (secondes)')
    ax1.set_title('Comparaison: Inversion Directe')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    #ax1.set_yscale('log')
    
    # Graphique 2: Inversion via Cholesky
    ax2.plot(P_values, results['inversion_chol_global'], 
             label='Cholesky global (NP×NP)', marker='o', linewidth=2)
    ax2.plot(P_values, results['inversion_chol_blocks'], 
             label='Cholesky par blocs (P × N×N)', marker='s', linewidth=2)
    ax2.set_xlabel('Nombre de blocs P')
    ax2.set_ylabel('Temps total (secondes)')
    ax2.set_title('Comparaison: Inversion via Cholesky')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    #ax2.set_yscale('log')
    
    # Graphique 3: Formes quadratiques directes
    ax3.plot(P_values, results['quad_global'], 
             label='Y^T @ K^-1 @ Y (global)', marker='o', linewidth=2)
    ax3.plot(P_values, results['quad_blocks'], 
             label='Σ y^T @ k^-1 @ y (blocs)', marker='s', linewidth=2)
    ax3.set_xlabel('Nombre de blocs P')
    ax3.set_ylabel('Temps total (secondes)')
    ax3.set_title('Comparaison: Formes Quadratiques (Inversion Directe)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    #ax3.set_yscale('log')
    
    # Graphique 4: Formes quadratiques via Cholesky
    ax4.plot(P_values, results['quad_chol_global'], 
             label='Y^T @ K^-1 @ Y (Cholesky global)', marker='o', linewidth=2)
    ax4.plot(P_values, results['quad_chol_blocks'], 
             label='Σ y^T @ k^-1 @ y (Cholesky blocs)', marker='s', linewidth=2)
    ax4.set_xlabel('Nombre de blocs P')
    ax4.set_ylabel('Temps total (secondes)')
    ax4.set_title('Comparaison: Formes Quadratiques (Cholesky)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    #ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

# ============= EXÉCUTION DU BENCHMARK =============

if __name__ == "__main__":
    # Configuration du benchmark
    N = 100  # Taille de chaque bloc
    P_values = range(5, 10)  # Nombre de blocs à tester
    n_repeats = 500  # Nombre de répétitions pour chaque mesure
    
    print("Démarrage du benchmark de comparaison matricielle pour GP...")
    print(f"Configuration: N={N}, P de {min(P_values)} à {max(P_values)}, {n_repeats} répétitions")
    
    # Exécution du benchmark
    results = run_benchmark(N=N, P_values=P_values, n_repeats=n_repeats)
    
    # Affichage des graphiques
    plot_results(results)
    
    # Analyse des résultats
    print("\n" + "="*60)
    print("ANALYSE DES RÉSULTATS:")
    print("="*60)
    
    # Point de bascule approximatif
    P_mid = len(P_values) // 2
    P_test = list(P_values)[P_mid]
    
    global_inv_time = results['inversion_chol_global_mean'][P_mid]
    blocks_inv_time = results['inversion_chol_blocks_mean'][P_mid]
    
    global_quad_time = results['quad_chol_global_mean'][P_mid]
    blocks_quad_time = results['quad_chol_blocks_mean'][P_mid]
    
    print(f"Pour P={P_test} (milieu de la plage):")
    print(f"  Inversion Cholesky - Global: {global_inv_time:.4f}s, Blocs: {blocks_inv_time:.4f}s")
    print(f"  Forme quadratique Cholesky - Global: {global_quad_time:.4f}s, Blocs: {blocks_quad_time:.4f}s")
    
    if blocks_inv_time < global_inv_time:
        print(f"  → L'approche par blocs est plus rapide pour l'inversion (gain: {global_inv_time/blocks_inv_time:.1f}x)")
    else:
        print(f"  → L'approche globale est plus rapide pour l'inversion (gain: {blocks_inv_time/global_inv_time:.1f}x)")
        
    if blocks_quad_time < global_quad_time:
        print(f"  → L'approche par blocs est plus rapide pour les formes quadratiques (gain: {global_quad_time/blocks_quad_time:.1f}x)")
    else:
        print(f"  → L'approche globale est plus rapide pour les formes quadratiques (gain: {blocks_quad_time/global_quad_time:.1f}x)")
    
    print("\nConclusion: L'approche par blocs est généralement plus efficace")
    print("pour les matrices ayant une structure bloc-diagonale, ce qui est")
    print("typique dans les GP multi-sorties avec noyaux indépendants.")