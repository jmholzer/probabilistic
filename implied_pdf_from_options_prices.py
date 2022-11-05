
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import splev, splrep
from diff_tvr import *

# %%
#--- clean call price data ---#

calls = pd.read_excel("SPY_311022exp_061022.xlsx", sheet_name="call")
#calls = calls.apply(pd.to_numeric)
calls["midprice"] = (calls.bid + calls.ask)/2
calls = calls[calls.midprice > 0]

# %%

def call_value(S, K, sigma, t=0, r=0):
    # use np.multiply and divide to handle divide-by-zero
    with np.errstate(divide='ignore'):
        d1 = np.divide(1, sigma * np.sqrt(t)) * (np.log(S/K) + (r+sigma**2 / 2) * t)
        d2 = d1 - sigma * np.sqrt(t)
    return np.multiply(norm.cdf(d1), S) - np.multiply(norm.cdf(d2), K * np.exp(-r * t))


def call_vega(S, K, sigma, t=0, r=0):
    with np.errstate(divide='ignore'):
        d1 = np.divide(1, sigma * np.sqrt(t)) * (np.log(S/K) + (r+sigma**2 / 2) * t)
    return np.multiply(S, norm.pdf(d1)) * np.sqrt(t)


def bs_iv(price, S, K, t=0, r=0, precision=1e-4, initial_guess=0.2, max_iter=1000, verbose=False):
    iv = initial_guess
    for _ in range(max_iter):
        P = call_value(S, K, iv, t, r)
        diff = price - P
        if abs(diff) < precision:
            return iv
        grad = call_vega(S, K, iv, t, r)
        iv += diff/grad
    if verbose:
        print(f"Did not converge after {max_iter} iterations")
    return iv

# %%
# setting parameters
S = 377
t = 3/52

# %%
# creating a column of implied volatility given the options data
calls["iv"] = calls.apply(lambda row: bs_iv(row.midprice, S, row.strike, t, max_iter=500), axis=1)

# %%
# drop cells where IV is NA
calls_no_na = calls.dropna()

#%%
# visualize the price-strike graph
calls_clean = calls.dropna().copy()
plt.scatter(calls_clean['strike'], calls_clean['midprice'])
plt.show()

# %%
# visualize the IV-strike graph
plt.scatter(calls_clean['strike'], calls_clean['iv'])
plt.show()

# %%
# FORK: put IV values through a gaussian filter
# there is no academic reason to do this
###--- the author's motivation to do this is ostensibly to generate a smoother first derivative --###
# as shown by the plot, the gaussian filtered IV will only be smoother if the initial data is very unsmooth. Smoothness of the gaussian filtered IV also depends on the smoothing parameter
  # if we were gonna do gaussian filtering, we'll need a way to tune the parameter automatically 
calls_clean["iv_gaussianfilter"] = gaussian_filter1d(calls_clean.iv, 3)

plt.scatter(calls_clean['strike'], calls_clean['iv_gaussianfilter'])
plt.show()

# %%
# FORK 2: using scipy's interp1d to interpolate continuous IV smile
# academic sources imply that a cubic spline is the standard choice

vol_surface = interp1d(calls_clean.strike, calls_clean.iv, kind="cubic", fill_value="extrapolate")
  # note that I chose to interpolate using the raw IV, without gaussian filtering

# %%
# plotting the IV smile

# create domain of final PDF
x_new = np.arange(calls_clean.strike.min(), calls_clean.strike.max(), 0.05)

plt.plot(calls_clean.strike, calls_clean.iv, "bx", x_new, vol_surface(x_new), "k-");
plt.legend(["smoothed IV", "fitted smile"], loc="best")
plt.xlabel("Strike")
plt.ylabel("IV")
plt.tight_layout()
# plt.savefig("SPY_smile.png", dpi=300)
plt.show()

# %%
# re-values call options using the BS formula, taking in as inputs S, domain, IV, and time to expiry
C_interp = call_value(S, x_new, vol_surface(x_new), t)

# %%
# plot the implied call prices
plt.scatter(x=x_new, y=C_interp, s=0.1)
plt.show()

# %%
"""
STRATEGY: 
WE DO THE NUMERICAL DIFFERENTIATION AS USUAL TO GET THE FIRST DERIVATIVE
THE FIRST DERIVATIVE WILL BE NOISY, SO WE USE TVR TO DIFFERENTIATE THIS NOISY DATA AND HOPEFULLY GET A SMOOTH 2ND DERIV

SO IGNORE THE FOLLOWING FEW CELLS WHERE TO USE TVR TO OBTAIN THE FIRST DERIVATIVE
"""
"""
#--- first derivative of the call options prices using TVR! ---#
n = len(C_interp)

# Derivative with TVR
# this is an optimization algorithm, so it will take a long time
# especially because our implied call-prices array is 2500 points of data - this took like 5 min
# we can consider fewer data points for faster calculation

ACTUALLY YES!! I think the reason why the author uses so many points of data is because he's doing numerical 
differentiation on the data points directly - therefore he needs a tiny 'dx'.

Since we're just essentially fitting a model on the noisy data, we don't need so many values 

diff_tvr = DiffTVR(n, dx)
(deriv, _) = diff_tvr.get_deriv_tvr(
    data=C_interp,
    deriv_guess=np.full(n + 1, 0.0),
    alpha=0.1,
    no_opt_steps=100
)

# %%
# create sparse C values (SEE NEXT CELL FOR REASON)
dx = 1
x_new_sparse = np.arange(calls_clean.strike.min(), calls_clean.strike.max(), dx)
# re-values call options using the BS formula, taking in as inputs S, domain, IV, and time to expiry
C_interp_sparse = call_value(S, x_new_sparse, vol_surface(x_new_sparse), t)

# plot the implied call prices
plt.scatter(x=x_new_sparse, y=C_interp_sparse, s=1)
plt.show()

# %%
#--- first derivative of the call options prices ---#
# using TVR!
n = len(C_interp_sparse)

# Derivative with TVR
# this is an optimization algorithm, so it will take a long time

diff_tvr = DiffTVR(n, dx)
(deriv_1, _) = diff_tvr.get_deriv_tvr(
    data=C_interp_sparse,
    deriv_guess=np.full(n + 1, 0.0),
    alpha=1,
    no_opt_steps=100
)

# %%
# show first derivative
plt.scatter(x=x_new_sparse, y=deriv_1[0:125])
plt.show()


# %%
first_deriv_discrete = np.gradient(C_interp, x_new)
# plot the first derivative
plt.scatter(x=x_new, y=first_deriv_discrete)
plt.show()

# %%
"""
# %%
#--- second derivative of the call options prices ---#
# using TVR!
n = len(C_interp)

# Derivative with TVR
# this is an optimization algorithm, so it will take a long time

diff_tvr = DiffTVR(n, 0.05)
(deriv_2, _) = diff_tvr.get_deriv_tvr(
    data=first_deriv_discrete,
    deriv_guess=np.full(n + 1, 0.0),
    alpha=10,
    no_opt_steps=100
)

print('done')

# %%
plt.scatter(x=x_new, y=deriv_2[0:2500], s=0.1)
plt.title('TVR 2nd derivative, alpha = 100')
plt.show()