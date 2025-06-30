import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import SamplingVQE, NumPyMinimumEigensolver, QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import Sampler
from qiskit_finance.data_providers import YahooDataProvider
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.converters import QuadraticProgramToQubo
import datetime
import pandas as pd
import yfinance as yf

# Simulate data for demonstration
algorithm_globals.random_seed = 42
stocks = ["AAPL", "AMZN", "MSFT", "META", "TSLA", "JPM"]

_real_download = yf.download
def fixed_download(*args, **kwargs):
    kwargs['auto_adjust'] = False
    return _real_download(*args, **kwargs)
yf.download = fixed_download

data = YahooDataProvider(
    tickers=stocks,
    start=datetime.datetime(2021, 1, 1),
    end=datetime.datetime(2021, 12, 31),
)
data.run()

mu = data.get_period_return_mean_vector()
sigma = data.get_period_return_covariance_matrix()

for (cnt, s) in enumerate(data._tickers):
    plt.plot(data._data[cnt], label=s)

plt.legend(loc="upper left")
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.title("Asset Prices (Raw Data)")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()

# === Generate Random Portfolios ===
num_portfolios = 5000
random_weights, random_returns, random_risks = [], [], []

for _ in range(num_portfolios):
    w = np.random.rand(len(stocks))
    w /= np.sum(w)
    random_weights.append(w)
    random_returns.append(mu @ w)
    random_risks.append(np.sqrt(w.T @ sigma @ w))

# === Efficient Frontier via Convex Optimization ===
target_returns = np.linspace(min(random_returns), max(random_returns), 100)
efficient_returns, efficient_risks = [], []

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

# === Classical Exact Optimizer ===
budget = 3
q = 0.5
po_classical = PortfolioOptimization(expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget)
qp = po_classical.to_quadratic_program()
exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
exact_result = exact_solver.solve(qp)
x_classical = exact_result.x
w_classical = x_classical / np.sum(x_classical)
ret_classical = mu @ w_classical
risk_classical = np.sqrt(w_classical.T @ sigma @ w_classical)

# === QAOA for Multiple λ ===
lambdas = [0.1, 1, 10, 100, 1000]
qaoa_points = []
optimizer = SPSA()
optimizer.set_options(maxiter=250)
ansatz = TwoLocal(len(stocks), "ry", "cz", reps=3, entanglement="full")
ansatz.decompose().draw(output='mpl')
qaoa_solver = QAOA(sampler=Sampler(), optimizer=optimizer, reps=3)

for lam in lambdas:
    po_qaoa = PortfolioOptimization(
        expected_returns=mu,
        covariances=sigma,
        risk_factor=q,
        budget=budget
    )
    qp = po_qaoa.to_quadratic_program()
    qubo_converter = QuadraticProgramToQubo(penalty=lam)
    qp_qaoa = qubo_converter.convert(qp)

    qaoa = MinimumEigenOptimizer(qaoa_solver)
    result_qaoa = qaoa.solve(qp_qaoa)

    x = result_qaoa.x
    if np.sum(x) > 0:
        w = x / np.sum(x)
        ret = mu @ w
        risk = np.sqrt(w.T @ sigma @ w)
        qaoa_points.append((risk, ret, lam))

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.scatter(random_risks, random_returns, color='lightgray', alpha=0.5, label='Random Portfolios')
plt.plot(efficient_risks, efficient_returns, color='blue', linewidth=3, label='Efficient Frontier')

# Classical
plt.scatter(risk_classical, ret_classical, marker='s', color='black', s=100, label='Classical Optimal')

# QAOA points
for risk, ret, lam in qaoa_points:
    plt.scatter(risk, ret, s=120, label=f'QAOA λ={lam}', marker='p')

plt.xlabel("Portfolio Risk (Volatility)")
plt.ylabel("Portfolio Return")
plt.title("Efficient Frontier with Classical and QAOA Solutions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Display QAOA Results ===
qaoa_df = pd.DataFrame({
    "Lambda": [lam for _, _, lam in qaoa_points],
    "QAOA Return": [ret for _, ret, _ in qaoa_points],
    "QAOA Risk": [risk for risk, _, _ in qaoa_points],
})
print(qaoa_df)