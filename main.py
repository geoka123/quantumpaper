from qiskit_finance import QiskitFinanceError
from qiskit_finance.data_providers import YahooDataProvider
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import SamplingVQE
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit.result import QuasiDistribution
import datetime
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import numpy as np
import yfinance as yf
import pandas as pd
import cvxpy as cp

# Force yfinance to return 'Adj Close'
_real_download = yf.download
def fixed_download(*args, **kwargs):
    kwargs['auto_adjust'] = False
    return _real_download(*args, **kwargs)
yf.download = fixed_download

# PARAMETERS
TRADING_DAYS = 252
stocks = ["MSFT", "AAPL", "GOOG", "AMZN"]
q = 0.5          # risk aversion
budget = 2       # number of assets to pick
penalty = 1

def print_result(result, po):
    selection = result.x
    value = result.fval
    print("Optimal: selection {}, value {:.4f}".format(selection, value))
    eigenstate = result.min_eigen_solver_result.eigenstate
    probabilities = (
        eigenstate.binary_probabilities()
        if isinstance(eigenstate, QuasiDistribution)
        else {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
    )
    print("\n----------------- Full result ---------------------")
    print("selection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for k, v in probabilities:
        x = np.array([int(i) for i in list(reversed(k))])
        value = po.to_quadratic_program().objective.evaluate(x)
        print("%10s\t%.4f\t\t%.4f" % (x, value, v))

# Load data
data = YahooDataProvider(
    tickers=stocks,
    start=datetime.datetime(2021, 1, 1),
    end=datetime.datetime(2021, 12, 31),
)
data.run()

# Annualize returns and covariance
mu = data.get_period_return_mean_vector() * TRADING_DAYS
sigma = data.get_period_return_covariance_matrix() * TRADING_DAYS

# Print stats
print(f"Mean returns:\n{mu}\n")
print(f"Covariance matrix:\n{sigma}\n")

# Build the portfolio problem
po = PortfolioOptimization(expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget)
qp = po.to_quadratic_program()

# Plot raw data
for (cnt, s) in enumerate(data._tickers):
    plt.plot(data._data[cnt], label=s)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3)
plt.xticks(rotation=90)
plt.title("Asset Prices (Raw Data)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Set up VQE
algorithm_globals.random_seed = 1234
optimizer = COBYLA()
optimizer.set_options(maxiter=500)
ansatz = TwoLocal(len(stocks), "ry", "cz", reps=3, entanglement='full')
ansatz.decompose().draw(output='mpl')
plt.show()

svqe_mes = SamplingVQE(sampler=Sampler(), ansatz=ansatz, optimizer=optimizer)
svqe = MinimumEigenOptimizer(svqe_mes)
result = svqe.solve(qp)


import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# === Generate Random Portfolios ===
num_portfolios = 5000
random_weights = []
random_returns = []
random_risks = []

for _ in range(num_portfolios):
    weights = np.random.rand(len(stocks))
    weights /= np.sum(weights)
    random_weights.append(weights)

    portfolio_return = mu @ weights
    portfolio_risk = np.sqrt(weights.T @ sigma @ weights)

    random_returns.append(portfolio_return)
    random_risks.append(portfolio_risk)

# === Compute Efficient Frontier ===
target_returns = np.linspace(min(random_returns), max(random_returns), 100)
efficient_returns = []
efficient_risks = []

for r_target in target_returns:
    w = cp.Variable(len(stocks))
    risk_expr = cp.quad_form(w, sigma)
    constraints = [cp.sum(w) == 1, mu @ w == r_target, w >= 0]
    problem = cp.Problem(cp.Minimize(risk_expr), constraints)
    
    try:
        problem.solve()
        if w.value is not None:
            efficient_returns.append(r_target)
            efficient_risks.append(np.sqrt(risk_expr.value))
    except:
        continue

# === Extract VQE solution ===
x_vqe = result.x  # binary selection vector from VQE
if np.sum(x_vqe) > 0:
    weights_vqe = x_vqe / np.sum(x_vqe)
    ret_vqe = mu @ weights_vqe
    risk_vqe = np.sqrt(weights_vqe.T @ sigma @ weights_vqe)

# === Plot Everything ===
plt.figure(figsize=(10, 6))

# Random portfolios (gray)
plt.scatter(random_risks, random_returns, color='lightgray', alpha=0.5, label='Random Portfolios')

# Efficient Frontier (blue)
plt.plot(efficient_risks, efficient_returns, color='blue', linewidth=3, label='Efficient Frontier')

# Max Sharpe Portfolio (red star)
rf = 0.01  # risk-free rate
sharpe_ratios = (np.array(random_returns) - rf) / np.array(random_risks)
max_sharpe_idx = np.argmax(sharpe_ratios)
plt.scatter(random_risks[max_sharpe_idx], random_returns[max_sharpe_idx],
            marker='*', color='red', s=200, label='Max Sharpe')

# VQE solution (green square)
plt.scatter(risk_vqe, ret_vqe, marker='s', color='green', s=100, label='VQE Optimal')

# === Labels and Legend ===
plt.xlabel("Portfolio Risk (Volatility)")
plt.ylabel("Portfolio Return")
plt.title("Efficient Frontier with Random and VQE Portfolios")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print quantum result
print_result(result, po)
