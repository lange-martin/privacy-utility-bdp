import numpy as np
from math import comb, exp, log
import matplotlib.pyplot as plt

# ----------------------
# INPUT DEL USUARIO
# ----------------------
n = 500               # Número de usuarios
eps = 4           # Privacidad base
a = 0.3               # Valor proporcionado por el usuario (puedes cambiarlo)

# ----------------------
# PARÁMETROS DERIVADOS
# ----------------------
# Nuevo ε para el mecanismo Laplaciano
gamma = (1-a)/(a)
eps_laplace = eps - 4 * log(gamma)
scale_lap = 1 / (n * eps_laplace)

# Nuevo p para RR
rho = (4 + a * (a * np.exp(eps) - 2) - np.sqrt(a**2 * np.exp(eps) * (4 + a * (a * np.exp(eps) - 4)))) / (8 + 2 * a * (a * np.exp(eps) + a - 4))
p_rr = 1 - rho
scale_rr = 2 * p_rr - 1

thetas = [0.1, 0.5, 0.7]
alphas = np.linspace(0.0, 0.2, 200)

# ----------------------
# FUNCIONES AUXILIARES
# ----------------------

def binom_pmf(k, n, q):
    return comb(n, k) * (q ** k) * ((1 - q) ** (n - k))

def compute_beta_rr(alpha, q, scale, n):
    '''
    Compute probality vector Pr[Z=k] for k=0,...,n 
    where Z=Sum yi follwing Binomial(n,q)
    q=thetha*p+(1-theta)(1-p)
    '''
    pmf = np.array([binom_pmf(k, n, q) for k in range(n + 1)])
    '''
    Array with all possible reported means 0,1/n,...,1
    '''
    barY = np.arange(n + 1) / n
    '''
    Boolean vector mask indicate which values of barY 
    satisfied the derired error threshold.
    '''
    mask = np.abs(barY - q) >= scale * alpha
    '''
    Finally we sum the prob of those values to obtain beta
    '''
    return pmf[mask].sum()

def compute_beta_laplace(alpha, scale):
    '''beta=exp(-alpha*eps*N)'''
    return  np.exp(-alpha / scale)

def compute_beta_guess(alpha, n, true_val=0.5):
    """Beta for random guessing using Bin(n, 0.5)/n"""
    pmf = np.array([binom_pmf(k, n, 0.5) for k in range(n + 1)])
    barG = np.arange(n + 1) / n
    mask = np.abs(barG - true_val) > alpha
    return pmf[mask].sum()

# Accuracy of random guessing for true θ = 0.5 (worst case)
betas_guess = [compute_beta_guess(alpha, n, true_val=0.5) for alpha in alphas]


# ----------------------
# CÁLCULO DE CURVAS
# ----------------------

# Randomized Response
results_rr = {}
for theta in thetas:
    q = scale_rr * theta + (1 - p_rr)
    betas = [compute_beta_rr(alpha, q, scale_rr, n) for alpha in alphas]
    results_rr[theta] = list(zip(alphas, betas))

# Laplaciano
betas_laplace = [compute_beta_laplace(alpha, scale_lap) for alpha in alphas]

# ----------------------
# GRAFICAR Y GUARDAR
# ----------------------

plt.figure(figsize=(5, 5))
for theta in thetas:
    alpha_vals, beta_vals = zip(*results_rr[theta])
    plt.plot(alpha_vals, beta_vals,linewidth=2.5,label=f"BDP RR, $\\theta = {theta:.1f}$")

plt.plot(alphas, betas_laplace,linewidth=2.5, label="Ours", color="black", linestyle="-")

plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.xlabel(r"$\alpha$", fontsize=16)
plt.ylabel(r"$\beta$", fontsize=16)
plt.title(f"$(\\alpha, \\beta)$-accuracy", fontsize=14)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

# Guardar imagen
plt.savefig(f"BDP_accuracy_eps{eps}_r{a}_n{n}.png", dpi=300)
print("Imagen guardada como 'BDP_accuracy_Laplace_vs_rr.png'")
