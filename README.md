# Reinforcement Learning for Portfolio Optimization using DDPG

## Overview

This project implements a Deep Deterministic Policy Gradient (DDPG) agent for portfolio optimization and compares its performance against traditional financial portfolio strategies.

## Environment and Agent

- **Environment**: A custom stock trading environment simulates portfolio dynamics with discrete daily stock prices.
- **Agent**: A DDPG agent learns to allocate shares across stocks to maximize portfolio value over time.
- **Training Data**: Historical price data (2016–2018) from selected Dow Jones stocks is fetched using `yfinance`.
- **Evaluation**: After training, the DDPG strategy is compared against:
  - Minimum Variance Portfolio
  - Dow Jones Index
- **Metrics**: Performance is evaluated using return, volatility, Sharpe/Sortino/Calmar ratios, drawdowns, and alpha/beta statistics.

## Key Components

- `StockTradingEnv`: Simulates portfolio rebalancing with stock price dynamics.
- `Actor` / `Critic`: Neural networks guiding the DDPG decision-making.
- `DDPGAgent`: Learns a policy to maximize returns via continuous action space.
- `train_agent()`: Trains the agent across multiple episodes and tracks portfolio value.
- `optimal_portfolio_weights()` / `min_var_portfolio_weights()`: Generate traditional portfolios via Sharpe ratio or minimum variance optimization.
- `generate_comparative_tear_sheet()`: Outputs a detailed performance comparison across strategies.

## How to Run

1. Install required packages:
   ```bash
   pip install numpy pandas yfinance matplotlib torch statsmodels scipy
