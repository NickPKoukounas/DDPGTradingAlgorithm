import numpy as np
import pandas as pd
from datetime import date
import datetime as dt
from dateutil.relativedelta import relativedelta
import torch
import torch.nn as nn
import scipy.optimize as sco
import warnings
import statsmodels.api as sm
import torch.optim as optim
import yfinance as yf
from copy import deepcopy
import matplotlib.pyplot as plt


#Stock Trading Environment
class StockTradingEnv:
    def __init__(self, stock_data, initial_balance=10000):
        self.stock_data = stock_data  #DataFrame with daily prices for stocks
        self.num_stocks = stock_data.shape[1]
        self.initial_balance = initial_balance
        self.weights_history = []
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        
        self.holdings = np.zeros(self.num_stocks)  #Number of shares per stock
        self.current_step = 0
        self.portfolio_value = self.balance
        return self._get_state()

    def _get_state(self):
        prices = self.stock_data.iloc[self.current_step].values
        return np.concatenate([prices, self.holdings, [self.balance]])
    
    def _store_weights(self):
        """Store current portfolio weights"""
        prices = self.stock_data.iloc[self.current_step].values
        stock_values = self.holdings * prices
        total_value = self.portfolio_value
        if total_value > 0:
            stock_weights = stock_values / total_value
            cash_weight = self.balance / total_value
            weights = np.concatenate([stock_weights, [cash_weight]])
        else:
            weights = np.zeros(self.num_stocks + 1)  #All zeros if no value
        self.weights_history.append(weights)

    def step(self, action):
        prices = self.stock_data.iloc[self.current_step].values
        prev_portfolio_value = self.balance + np.sum(prices * self.holdings)

        #Action: array of buy/sell/hold for each stock (negative for sell, positive for buy)
        for i, act in enumerate(action):
            act = int(np.clip(act, -self.holdings[i], (self.balance / prices[i])))  #Limit by balance/holdings
            if act > 0:  # Buy
                cost = act * prices[i]
                if cost <= self.balance:
                    self.holdings[i] += act
                    self.balance -= cost
            elif act < 0:  
                self.holdings[i] += act  #act is negative
                self.balance -= act * prices[i]  #Negative act * price = positive cash

        self.current_step += 1
        new_prices = self.stock_data.iloc[self.current_step].values if self.current_step < len(self.stock_data) else prices
        self.portfolio_value = self.balance + np.sum(new_prices * self.holdings)
        reward = self.portfolio_value - prev_portfolio_value
        self._store_weights()

        done = self.current_step >= len(self.stock_data) - 1
        return self._get_state(), reward, done

#Neural Network Models
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() 
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))
    

#DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action=100): #max action # Shares of stock per time step
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005
        self.noise_scale = 0.1

    def act(self, state, noise=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        if noise:
            action += np.random.normal(0, self.noise_scale, size=action.shape)
        return np.clip(action * self.max_action, -self.max_action, self.max_action)

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        #Critic update
        next_actions = self.actor_target(next_states)
        target_q = self.critic_target(next_states, next_actions)
        y = rewards + (1 - dones) * self.gamma * target_q
        critic_loss = nn.MSELoss()(self.critic(states, actions), y.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #Actor update
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #Update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



#Main loop
def train_agent(dow_tickers):
    #Fetch stock data (example with yfinance, replace with actual Dow 30 data)
        
     
    stock_data = yf.download(dow_tickers, start='2016-01-04', end='2018-09-28')['Close']
    env = StockTradingEnv(stock_data)

    state_dim = env.num_stocks * 2 + 1  #Prices, holdings, balance
    action_dim = env.num_stocks
    agent = DDPGAgent(state_dim, action_dim, max_action=100)  #Max shares per action

    episodes = 100
    portfolio_values = []
    


    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        env.weights_history = []
        daily_values = []
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            ep_reward += reward

            daily_values.append(env.portfolio_value)

        portfolio_values.append(env.portfolio_value)
        final_weights = env.weights_history[-1]
        stock_weights = final_weights[:-1]
        cash_weight = final_weights[-1]


        print(f"Episode {ep+1}/{episodes}, Portfolio Value: {env.portfolio_value:.2f}")
        #print(f"Stock Weights: {', '.join([f'{w:.3f}' for w in stock_weights])}, Cash Weight: {cash_weight:.3f}")
        
        
    return portfolio_values, daily_values

#Run

dow_tickers = [
    "AAPL",  # Apple
    "AXP",   # American Express
    "BA",    # Boeing
    "CAT",   # Caterpillar
    "CSCO",  # Cisco Systems
    "CVX",   # Chevron
    "DIS",   # Walt Disney
    "GE",    # General Electric (removed mid-2018, still valid if using early 2018)
    "GS",    # Goldman Sachs
    "HD",    # Home Depot
    "IBM",   # IBM
    "INTC",  # Intel
    
    "JPM",   # JPMorgan Chase
    "KO",    # Coca-Cola
    "MCD",   # McDonald's
    "MMM",   # 3M
  
    "MSFT",  # Microsoft
    "NKE",   # Nike
    "PFE",   # Pfizer
   
    "TRV",   # Travelers
    "UNH",   # UnitedHealth Group
    
    "V",     # Visa
    "VZ",    # Verizon
    "WMT",   # Walmart
    "XOM"    # ExxonMobil
]

portfolio_values, daily_values = train_agent(dow_tickers)

plt.plot(daily_values)
plt.show()
daily_values

df = pd.DataFrame(daily_values, columns=["Values"])
df.to_csv("DailyValues.csv", index=False)

DailyValRL = pd.read_csv("RL_portfolio_values.csv")
DailyValRL['Date'] = pd.to_datetime(DailyValRL['Date'])
DailyValRL = DailyValRL.dropna(subset=['Values'])
DailyValRL = DailyValRL.set_index('Date')
DailyValRL = DailyValRL.sort_index()
DailyValRL = DailyValRL['Values'].astype(float)
DailyValRL

# Start and end
train_start_date = "2013-06-04"
train_end_date = "2019-12-31"

# Test Start Date
test_start_date = '2016-01-01'
test_end_date = '2018-09-28'

# Date today
date_today = date.today().strftime('%Y-%m-%d')

# Risk-free rate
rf = 0

# Initial portfolio funding
initial_funding = 10000

# Tickers for the DJIA portfolio
tickers = [
    "MMM",  # 3M Company
    "AXP",  # American Express Company
    "AAPL", # Apple Inc.
    "BA",   # Boeing Company
    "CAT",  # Caterpillar Inc.
    "CVX",  # Chevron Corporation
    "CSCO", # Cisco Systems, Inc.
    "KO",   # Coca-Cola Company
    "DD",   # DuPont
    "XOM",  # Exxon Mobil Corporation
    "GE",   # General Electric Company
    "GS",   # Goldman Sachs Group, Inc.
    "HD",   # Home Depot, Inc.
    "INTC", # Intel Corporation
    "IBM",  # International Business Machines Corporation
    "JNJ",  # Johnson & Johnson
    "JPM",  # JPMorgan Chase & Co.
    "MCD",  # McDonald's Corporation
    "MRK",  # Merck & Co., Inc.
    "MSFT", # Microsoft Corporation
    "NKE",  # Nike, Inc.
    "PFE",  # Pfizer Inc.
    "PG",   # Procter & Gamble Company
    "TRV",  # Travelers Companies, Inc.
    "UNH",  # UnitedHealth Group Incorporated
    "RTX",  # Raytheon Technologies Corp.
    "VZ",   # Verizon Communications Inc.
    "V",    # Visa Inc.
    "WMT",  # Wal-Mart Stores, Inc.
    "DIS"   # Walt Disney Company
]


def optimal_portfolio_weights(ticker_list, start_date, end_date, rf=0.0):
    """
    Calculates the optimal portfolio weights for a list of tickers based on maximizing the Sharpe ratio.

    Parameters:
    - ticker_list (list): List of stock tickers (strings).
    - start_date (str): Start date for historical data (format: 'YYYY-MM-DD').
    - end_date (str): End date for historical data (format: 'YYYY-MM-DD').
    - rf (float): Risk-free rate (default 0.0, assuming no risk-free rate).

    Returns:
    - np.ndarray: Optimal portfolio weights (numpy array).
    """
    # Download adjusted close prices and compute daily returns
    returns = yf.download(ticker_list, start=start_date, end=end_date, auto_adjust=True)["Close"].dropna().pct_change().dropna()

    # Compute annualized return and covariance matrix
    mu = 252 * returns.mean().to_numpy()
    cov_matrix = 252 * returns.cov().to_numpy()

    # Define the Sharpe ratio objective function (we maximize it by minimizing its negative)
    def sharpe_ratio(weights):
        port_return = np.dot(weights, mu)
        port_volatility = np.sqrt(weights @ cov_matrix @ weights)
        return -(port_return - rf) / port_volatility  # Negative for minimization

    # Constraints: weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Bounds: each weight must be non-negative (long-only constraint)
    bounds = [(0, 1) for _ in range(len(ticker_list))]

    # Initial guess: equal allocation
    x0 = np.ones(len(ticker_list)) / len(ticker_list)

    # Solve the optimization problem
    result = sco.minimize(sharpe_ratio, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    # Return optimal portfolio weights
    return result.x

def min_var_portfolio_weights(ticker_list, start_date, end_date, rf=0.0):
    """
    Calculates the minimum variance portfolio weights for a list of tickers

    Parameters:
    - ticker_list (list): List of stock tickers (strings).
    - start_date (str): Start date for historical data (format: 'YYYY-MM-DD').
    - end_date (str): End date for historical data (format: 'YYYY-MM-DD').
    - rf (float): Risk-free rate (default 0.0, assuming no risk-free rate).

    Returns:
    - np.ndarray: Optimal portfolio weights (numpy array).
    """
    # Download adjusted close prices and compute daily returns
    returns = yf.download(ticker_list, start=start_date, end=end_date, auto_adjust=True)["Close"].dropna().pct_change().dropna()

    # Compute annualized return and covariance matrix
    mu = 252 * returns.mean().to_numpy()
    cov_matrix = 252 * returns.cov().to_numpy()

    # Define the Sharpe ratio objective function (we maximize it by minimizing its negative)
    def portfolio_vol(weights):
        return np.sqrt(weights @ cov_matrix @ weights)  # Negative for minimization

    # Constraints: weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Bounds: each weight must be non-negative (long-only constraint)
    bounds = [(-1, 1) for _ in range(len(ticker_list))]

    # Initial guess: equal allocation
    x0 = np.ones(len(ticker_list)) / len(ticker_list)

    # Solve the optimization problem
    result = sco.minimize(portfolio_vol, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    # Return optimal portfolio weights
    return result.x

def calculate_max_drawdown_and_duration(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    drawdown_periods = drawdown < 0
    longest_drawdown = (drawdown_periods.astype(int)
                        .groupby((~drawdown_periods).astype(int).cumsum())
                        .cumsum()).max()

    return max_drawdown, longest_drawdown

def calculate_sortino_ratio(returns, risk_free=0.02):
    """Annualized Sortino Ratio."""
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    annualized_return = ((1 + returns).prod()) ** (252 / len(returns)) - 1
    if downside_std == 0:
        return np.nan
    return (annualized_return - risk_free) / (downside_std * np.sqrt(252))

def generate_comparative_tear_sheet(portfolio_list, portfolio_names, risk_free=0.02, benchmark_ticker='SPY'):
    start_date = min([pf.index[0] for pf in portfolio_list])
    end_date = max([pf.index[-1] for pf in portfolio_list])
    benchmark = yf.download(benchmark_ticker, start=start_date, end=end_date)['Close'].pct_change().dropna().squeeze()

    metrics = ['Cumulative Return', 'Annualized Return', 'Annualized Volatility',
               'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
               'Maximum Drawdown', 'Longest Drawdown Duration (days)',
               'Beta', 'Alpha']

    table_data = {'Metric': metrics}

    # Benchmark Metrics
    BM_daily_vol = np.sqrt(np.var(benchmark))
    BM_annual_vol = BM_daily_vol * np.sqrt(252)
    BM_Cum_Return = ((benchmark + 1).prod() - 1)
    BM_Annualized_Return = ((benchmark + 1).prod() ** (252 / len(benchmark)) - 1)
    BM_Sharpe_Ratio = (BM_Annualized_Return - risk_free) / BM_annual_vol
    BM_max_drawdown, BM_longest_drawdown_days = calculate_max_drawdown_and_duration(benchmark)
    BM_Sortino = calculate_sortino_ratio(benchmark, risk_free)
    BM_Calmar = BM_Annualized_Return / abs(BM_max_drawdown) if BM_max_drawdown != 0 else np.nan

    table_data[benchmark_ticker] = [
        BM_Cum_Return, BM_Annualized_Return, BM_annual_vol,
        BM_Sharpe_Ratio, BM_Sortino, BM_Calmar,
        BM_max_drawdown, BM_longest_drawdown_days,
        1.0, 0.0
    ]

    for portfolio, name in zip(portfolio_list, portfolio_names):
        returns = portfolio.pct_change().dropna().squeeze()
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]

        portfolio_daily_vol = np.sqrt(np.var(returns))
        portfolio_annual_vol = portfolio_daily_vol * np.sqrt(252)
        portfolio_Cum_Return = ((returns + 1).prod() - 1)
        portfolio_Annualized_Return = ((returns + 1).prod() ** (252 / len(returns)) - 1)
        portfolio_Sharpe_Ratio = (portfolio_Annualized_Return - risk_free) / portfolio_annual_vol
        portfolio_max_drawdown, portfolio_longest_drawdown_days = calculate_max_drawdown_and_duration(returns)
        portfolio_Sortino = calculate_sortino_ratio(returns, risk_free)
        portfolio_Calmar = portfolio_Annualized_Return / abs(portfolio_max_drawdown) if portfolio_max_drawdown != 0 else np.nan

        df = pd.DataFrame({benchmark_ticker: benchmark, name: returns}).dropna()
        X = sm.add_constant(df[benchmark_ticker])
        Y = df[name]
        model = sm.OLS(Y, X).fit()
        beta = model.params[benchmark_ticker]
        alpha = model.params['const']

        table_data[name] = [
            portfolio_Cum_Return, portfolio_Annualized_Return, portfolio_annual_vol,
            portfolio_Sharpe_Ratio, portfolio_Sortino, portfolio_Calmar,
            portfolio_max_drawdown, portfolio_longest_drawdown_days,
            beta, alpha
        ]

    result_df = pd.DataFrame(table_data)
    result_df.set_index('Metric', inplace=True)

    try:
        display(result_df.style.set_table_styles(
            [{'selector': 'thead th', 'props': [('font-weight', 'bold'), ('text-align', 'center'),
                                                ('background-color', 'black'), ('color', 'white')]},
             {'selector': 'tbody td', 'props': [('text-align', 'center')]},
             {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%')]},
             {'selector': 'tr', 'props': [('border-bottom', '1px solid #ddd')]},
             {'selector': 'td', 'props': [('padding', '10px')]}]
        ))
    except NameError:
        print(result_df)

    return result_df


eq_weights = np.array([1/len(tickers)]*len(tickers))
min_var_weights=min_var_portfolio_weights(ticker_list=tickers,start_date=train_start_date,end_date=train_end_date)
tangent_weights = optimal_portfolio_weights(ticker_list=tickers,start_date=train_start_date,end_date=train_end_date)


eq_weight_portfolio_values = (yf.download(tickers, start=test_start_date, end=test_end_date, auto_adjust=True)['Close'].dropna()/yf.download(tickers, start=test_start_date, end=test_end_date, auto_adjust=True)['Close'].dropna().iloc[0] * eq_weights * initial_funding).sum(axis=1)
min_var_portfolio_values = (yf.download(tickers, start=test_start_date, end=test_end_date, auto_adjust=True)['Close'].dropna()/yf.download(tickers, start=test_start_date, end=test_end_date, auto_adjust=True)['Close'].dropna().iloc[0] * min_var_weights * initial_funding).sum(axis=1)
tangent_portfolio_values = (yf.download(tickers, start=test_start_date, end=test_end_date, auto_adjust=True)['Close'].dropna()/yf.download(tickers, start=test_start_date, end=test_end_date, auto_adjust=True)['Close'].dropna().iloc[0] * tangent_weights * initial_funding).sum(axis=1)
dow_jones_portfolio_values = (yf.download("^DJI", start=test_start_date, end=test_end_date, auto_adjust=True)['Close'].dropna()/yf.download("^DJI", start=test_start_date, end=test_end_date, auto_adjust=True)['Close'].dropna().iloc[0] * initial_funding)
iShares_Dow_portfolio_values = (yf.download("IYY", start=test_start_date, end=test_end_date, auto_adjust=True)['Close'].dropna()/yf.download("IYY", start=test_start_date, end=test_end_date, auto_adjust=True)['Close'].dropna().iloc[0] * initial_funding)

plt.figure(figsize=(12, 6))  # make chart wider
plt.plot(min_var_portfolio_values, label='Minimum Variance Portfolio')
#plt.plot(eq_weight_portfolio_values, label='Equal Weight Portfolio')
#plt.plot(tangent_portfolio_values, label='Tangent Portfolio')
plt.plot(dow_jones_portfolio_values, label='Dow Jones Industrial Average')
#plt.plot(iShares_Dow_portfolio_values, label='iShares Dow ETF')
plt.plot(DailyValRL, label='DDPG')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.title('Benchmarks V.S. Reinforcement Learning Portfolio')
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()  # Automatically format the x-axis dates
plt.tight_layout()         # Adjust layout to avoid clipping
plt.show()
