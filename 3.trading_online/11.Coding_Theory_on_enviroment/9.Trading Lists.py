#!/usr/bin/env python
# coding: utf-8

# # Trading Lists and Trading Trajectories
# 
# ### Introduction
# 
# [Almgren and Chriss](https://cims.nyu.edu/~almgren/papers/optliq.pdf) provided a solution to the optimal liquidation problem by assuming the that stock prices follow a discrete arithmetic random walk, and that the permanent and temporary market impact functions are linear functions of the trading rate.
# 
# Almgren and Chriss showed that for each value of risk aversion there is a unique optimal execution strategy. This optimal execution strategy is determined by a trading trajectory and its associated trading list. The optimal trading trajectory is given by:
# 
# \begin{equation}
# x_j = \frac{\sinh \left( \kappa \left( T-t_j\right)\right)}{ \sinh (\kappa T)}X, \hspace{1cm}\text{ for } j=0,...,N
# \end{equation}
# 
# and the associated trading list is given by:
# 
# \begin{equation}
# n_j = \frac{2 \sinh \left(\frac{1}{2} \kappa \tau \right)}{ \sinh \left(\kappa T\right) } \cosh \left(\kappa \left(T - t_{j-\frac{1}{2}}\right)\right) X, \hspace{1cm}\text{ for } j=1,...,N
# \end{equation}
# 
# where $t_{j-1/2} = (j-\frac{1}{2}) \tau$.
# 
# Given some initial parameters, such as the number of shares, the liquidation time, the trader's risk aversion, etc..., the trading list will tell us how many shares we should sell at each trade to minimize our transaction costs. 
# 
# In this notebook, we will see how the trading list varies according to some initial trading parameters. 
# 
# ## Visualizing Trading Lists and Trading Trajectories
# 
# Let's assume we have 1,000,000 shares that we wish to liquidate. In the code below, we will plot the optimal trading trajectory and its associated trading list for different trading parameters, such as trader's risk aversion, number of trades, and liquidation time. 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

import utils

# We set the default figure size
plt.rcParams['figure.figsize'] = [17.0, 7.0]


# Set the number of days to sell all shares (i.e. the liquidation time)
l_time = 60

# Set the number of trades
n_trades = 60

# Set the trader's risk aversion
t_risk = 1e-6

# Plot the trading list and trading trajectory. If show_trl = True, the data frame containing the values of the
# trading list and trading trajectory is printed
utils.plot_trade_list(lq_time = l_time, nm_trades = n_trades, tr_risk = t_risk, show_trl = True)


# # Implementing a Trading List
# 
# Once we have the trading list for a given set of initial parameters, we can actually implement it. That is, we can sell our shares in the stock market according to the trading list and see how much money we made or lost. To do this, we are going to simulate the stock market with a simple trading environment. This simulated trading environment uses the same price dynamics and market impact functions as the Almgren and Chriss model. That is, stock price movements evolve according to a discrete arithmetic random walk and the permanent and temporary market impact functions are linear functions of the trading rate. We are going to use the same environment to train our Deep Reinforcement Learning algorithm later on.
# 
# We will describe the details of the trading environment in another notebook, for now we will just take a look at its default parameters. We will distinguish between financial parameters, such the annual volatility in stock price, and the parameters needed to calculate the trade list using the Almgren and Criss model, such as the trader's risk aversion.

# In[ ]:


import utils

# Get the default financial and AC Model parameters
financial_params, ac_params = utils.get_env_param()


# ### Default Financial Parameters

# In[ ]:


financial_params


# ### Parameters for the Almgren and Chriss Model

# In[ ]:


ac_params


# The code below implements the trading list resulting from different trading parameters, such as trader's risk aversion, number of trades, and liquidation time. All other parameters, such as total number shares to sell, are taken from the simulated trading environment (see above). 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

import utils

# We set the default figure size
plt.rcParams['figure.figsize'] = [17.0, 7.0]


# Set the random seed
sd = 0

# Set the number of days to sell all shares (i.e. the liquidation time)
l_time = 60

# Set the number of trades
n_trades = 60

# Set the trader's risk aversion
t_risk = 1e-6

# Implement the trading list for the given parameters
utils.implement_trade_list(seed = sd, lq_time = l_time, nm_trades = n_trades, tr_risk = t_risk)


# In[ ]:




