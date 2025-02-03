import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf

# Manually override pandas_datareader to use yfinance
def get_data(stocks, start, end):
    try:
        stockData = yf.download(stocks, start=start, end=end)  # Using yfinance directly
        stockData = stockData['Close']  # Fixing the typo
        returns = stockData.pct_change()
        meanReturns = returns.mean()
        covMatrix = returns.cov()
        return meanReturns, covMatrix
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None

# Define stock symbols and date range
stocklist = ['RY', 'TD', 'BMO', 'BNS', 'CM']
stocks = [stock + '.TO' for stock in stocklist]
enddate = dt.datetime.now()
startdate = enddate - dt.timedelta(days=300)

# Fetch data
meanReturns, covMatrix = get_data(stocks, startdate, enddate)

# Check the result
if meanReturns is not None:
    print(meanReturns)
else:
    print("Data could not be fetched.")
    
weights = np.random.random(len(meanReturns))

weights /= np.sum(weights)
print(weights)



# Monte Carlo Simulation Parameters
mc_sim = 5  # Number of simulations
T = 10  # Number of trading days
init_portfolio = 10000  # Starting amount

# Reshape mean returns correctly
meanMatrix = np.tile(meanReturns.values, (T, 1))  # (T, number of stocks)

# Initialize simulation matrix
portfolio_sim = np.zeros((T, mc_sim))

for m in range(mc_sim):
    Z = np.random.normal(size=(T, len(weights)))  # Random noise
    L = np.linalg.cholesky(covMatrix)  # Cholesky decomposition
    daily_ret = meanMatrix + np.dot(Z, L.T)  # Correct return simulation
    cumulative_returns = np.cumprod(1 + np.dot(daily_ret, weights))  # Calculate portfolio growth
    portfolio_sim[:, m] = init_portfolio * cumulative_returns  # Scale by initial portfolio

# Compute the Mean Portfolio Path Across Simulations
mean_portfolio = portfolio_sim.mean(axis=1)

# Plot Results with Labels & Colors
plt.figure(figsize=(10,5))

# Plot each Monte Carlo path with transparency
for m in range(mc_sim):
    plt.plot(portfolio_sim[:, m], color='blue', alpha=0.2)  # Light blue lines for simulations

# Highlight the mean portfolio value in bold
plt.plot(mean_portfolio, color='red', linewidth=2, label="Mean Portfolio Value")  # Red line for mean path

# Add Labels and Title
plt.ylabel('Portfolio Value (CAD)')
plt.xlabel('Trading Days')
plt.title('Simulated Order of Returns')

# Add Legend
plt.legend()

# Show plot
plt.show()