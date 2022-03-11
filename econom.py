import pandas as pd
import numpy as np
from scipy.optimize import minimize
import time


x_submit = pd.read_csv("x_submit.csv", index_col="Date")
x_train = pd.read_csv("x_train_init.csv")
x_valid = pd.read_csv("x_valid_init.csv")

# def calc_diversification_ratio(w, V):
#     # average weighted vol
#     w_vol = np.dot(np.sqrt(np.diag(V)), w.T)
#     # portfolio vol
#     port_vol = np.sqrt(calculate_portfolio_var(w, V))
#     diversification_ratio = w_vol/port_vol
#     # return negative for minimization problem (maximize = minimize -)
#     return -diversification_ratio


# #####################################################################
# #               PORTFOLIO Optimization functions                    #
# #####################################################################

# def max_div_port(w0, V, bnd=None, long_only=True):

#     cons = ({'type': 'eq', 'fun': total_weight_constraint},)
    
#     if long_only: # add in long only constraint
#         cons = cons + ({'type': 'ineq', 'fun':  long_only_constraint},)
        
#     res = minimize(calc_diversification_ratio, w0, bounds=bnd, args=V, method='SLSQP', constraints=cons)
    
#     return res

returns = pd.pivot_table(x_train, index="Sousjacent", columns="Date", 
                         values="Y", fill_value=0)
r = returns.mean(axis=1)
w = np.ones(len(r))
V = np.cov(returns, rowvar=1)
vol = np.diag(V)**.5

def calc_div(w, V, vol):
    return -(np.dot(w, vol))/(np.dot(np.dot(w, V), w)**0.5)
""
def max_div_port(r, V, vol):
    x0 = np.ones(len(r))
    bounds = [(-0.01, 0.01) for _ in range(len(r))]
    cons = {'type': 'eq', 'fun':  lambda w: w.sum() - 1}
    calc_div_ptf = (lambda w: calc_div(w, V, vol))
    res = minimize(fun=calc_div_ptf,
                   x0=x0,
                   constraints=cons,
                   bounds=bounds)
    return res



# t = time.time()

# weights = max_div_port(r, V, vol)
# stocks = returns.index

# print(time.time() - t)

# stocknames = returns.index
# w = weights.x


isPresent = pd.pivot_table(x_valid, index="Sousjacent", columns="Date", 
                           values="Presence", fill_value=0)

# df_weights = pd.Series(index=stocknames, data=w).reindex(isPresent.index)

# for stockName in isPresent.index:
#     isPresent.loc[stockName] *= df_weights[stockName]

# for date in isPresent.columns:
#     longStocks = isPresent[date].nlargest(100).index
#     shortStocks = isPresent[date].nsmallest(100).index
#     x_submit.loc[date, longStocks] = 0.01
#     x_submit.loc[date, shortStocks] = -0.01

# x_submit.to_csv("Test_maxdiv.csv")





