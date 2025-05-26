import numpy as np
from math import comb, exp, log
import matplotlib.pyplot as plt

# ----------------------
# INPUT 
# ----------------------
n = 731               # Número de usuarios              # Privacidad base
a = 0.27               # Valor proporcionado por el usuario (puedes cambiarlo)
beta_target = 0.05    # Beta fijado

# ----------------------
# PARAMETERS
# ----------------------
gamma = (1 - a) / a
thetas = [0.1,0.5,0.7]
# ----------------------
#  AUXILARY FUNCTIONS
# ----------------------

def binom_pmf(k, n, q):
    return comb(n, k) * (q ** k) * ((1 - q) ** (n - k))

def compute_beta_rr(alpha, q, scale, n):
    pmf = np.array([binom_pmf(k, n, q) for k in range(n + 1)])
    barY = np.arange(n + 1)
    mask = np.abs(barY - q*n) >= scale * alpha
    return pmf[mask].sum()

def compute_beta_laplace(alpha, scale):
    return np.exp(-alpha / scale)

def find_alpha_for_beta_rr(beta, q, scale, n, alpha_max=1000, tol=1e-5):
    # Busca alpha tal que compute_beta_rr(alpha) == beta
    from scipy.optimize import bisect
    def f(alpha):
        return compute_beta_rr(alpha, q, scale, n) - beta
    #print(f(0),f(alpha_max))
    return bisect(f, 0, alpha_max, xtol=tol)

def find_alpha_for_beta_laplace(beta, scale):
    # beta = exp(-alpha/scale) => alpha = -scale * ln(beta)
    return -scale * np.log(beta)


# ----------------------
# PLOT ALPHA vs EPS 
# ----------------------

eps_vals = np.linspace(4, 20, 100)
alphas_rr_vs_eps = {theta: [] for theta in thetas}
alphas_laplace_vs_eps = []

for eps_val in eps_vals:
    # Recalculo parámetros para cada eps_val
    rho = (4 + a * (a * np.exp(eps_val) - 2) - np.sqrt(a**2 * np.exp(eps_val) * (4 + a * (a * np.exp(eps_val) - 4)))) / (8 + 2 * a * (a * np.exp(eps_val) + a - 4))
    p_rr = 1 - rho
    scale_rr = 2 * p_rr - 1
    gamma = (1-a) / a
    eps_laplace_val = eps_val - 4 * log(gamma)
    scale_lap_val = 1 / eps_laplace_val

    for theta in thetas:
        q = scale_rr * theta + (1 - p_rr)
        alpha_rr = find_alpha_for_beta_rr(beta_target, q, scale_rr, n)
        alphas_rr_vs_eps[theta].append(alpha_rr)

    alpha_laplace_val = find_alpha_for_beta_laplace(beta_target, scale_lap_val)
    alphas_laplace_vs_eps.append(alpha_laplace_val)

plt.figure(figsize=(6,5))
plt.yscale('log')
for theta in thetas:
    plt.plot(eps_vals, alphas_rr_vs_eps[theta], label=f"BDP RR $\\theta={theta}$", linewidth=2.5)
plt.plot(eps_vals, alphas_laplace_vs_eps, label="Laplace (Ours)", color="black", linestyle="-", linewidth=2.5)

plt.xlabel(r"$\epsilon$", fontsize=16)
plt.ylabel(r"$\alpha$", fontsize=16)
plt.title("Accuracy $\\alpha$ vs Privacy $\\epsilon$ (fixed $\\beta=0.05$)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=18)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()

plt.savefig(f"accu_counting_{beta_target}_n{n}_r{a}.png", dpi=300)
print(f"Plot saved as alpha_vs_eps_beta{beta_target}_n{n}_r{a}.png")
plt.show()
