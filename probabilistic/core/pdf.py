from typing import Tuple

import numpy as np
from pandas import DataFrame
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.integrate import simps

from probabilistic.core.tvr import DiffTVR


def calculate_pdf(
    options_data: DataFrame, current_price: float, days_forward: int
) -> Tuple[np.array]:
    """The main execution path for the pdf module. Takes a `DataFrame` of
    options data as input and makes a series of function calls to

    Args:
        options_data: a DataFrame containing options price data with
            cols ['strike', 'bid', 'ask']
        current_price: the current price of the security
        days_forward: the number of days in the future to estimate the
            price probability density at

    Returns:
        a tuple containing the price and density values (in numpy arrays)
        of the calculated PDF
    """
    options_data = _calculate_mid_price(options_data)
    options_data = _calculate_IV(options_data, current_price, days_forward)
    return _create_pdf_point_arrays(options_data, current_price, days_forward)


def calculate_cdf(pdf_point_arrays: Tuple[np.array]) -> Tuple[np.array]:
    """Returns the cumulative probability at each price. Takes as input the array
    of pdf and array of prices, and calculates the cumulative probability as the
    numerical integral over the pdf function.

    For simplicity, it assumes that the CDF at the starting price
        = 1 - 0.5*(total area of the pdf)
    and therefore it adds 0.5*(total area of the pdf) to every cdf for the
    remainder of the domain

    Args:
        pdf_point_arrays: a tuple containing the np.array of the calculated
        cumulative densities at each price and a np.array of the domain of prices
    """
    x_array, pdf_array = pdf_point_arrays
    cdf = []
    n = len(x_array)

    total_area = simps(y=pdf_array[0:n], x=x_array)
    remaining_area = 1 - total_area

    for i in range(n):
        if i == 0:
            integral = 0.0 + remaining_area / 2
        else:
            integral = simps(y=pdf_array[i - 1:i + 1], x=x_array[i - 1:i + 1]) + cdf[-1]
        cdf.append(integral)
    return [x_array, cdf]


def _calculate_mid_price(options_data: DataFrame) -> DataFrame:
    """Calculate the mid-price of the options at each strike price.

    Args:
        options_data: a DataFrame containing options price data with
            cols ['strike', 'bid', 'ask']

    Returns:
        the options_data DataFrame, with an additional column for mid-price
    """
    options_data["mid_price"] = (options_data.bid + options_data.ask) / 2
    options_data = options_data[options_data.mid_price > 0]
    return options_data


def _calculate_IV(
    options_data: DataFrame, current_price: float, days_forward: int
) -> DataFrame:
    """Calculate the implied volatility of the options in options_data

    Args:
        options_data: a DataFrame containing options price data with
            cols ['strike', 'bid', 'ask', 'mid_price']
        current_price: the current price of the security
        days_forward: the number of days in the future to estimate the
            price probability density at

    Returns:
        the options_data DataFrame, with an additional column for implied volatility
    """
    years_forward = days_forward / 365
    options_data["iv"] = options_data.apply(
        lambda row: _bs_iv(
            row.mid_price, current_price, row.strike, years_forward, max_iter=500
        ),
        axis=1,
    )
    options_data = options_data.dropna()
    return options_data


def _create_pdf_point_arrays(
    options_data: DataFrame, current_price: float, days_forward: int
) -> Tuple[np.array]:
    """Create two arrays containing x- and y-axis values representing a calculated
    price PDF

    Args:
        options_data: a DataFrame containing options price data with
            cols ['strike', 'bid', 'ask', 'iv']
        current_price: the current price of the security
        days_forward: the number of days in the future to estimate the
            price probability density at

    Returns:
        a tuple containing x-axis values (index 0) and y-axis values (index 1)
    """
    vol_surface = interp1d(
        options_data.strike, options_data.iv, kind="cubic", fill_value="extrapolate"
    )
    dx_1 = 0.05 # setting dx = 0.05 for the numerical differentiation of 1st derivative
    X = np.arange(options_data.strike.min(), options_data.strike.max(), dx_1)

    # re-values call options using the BS formula, taking in as inputs S, domain, IV, and time to expiry
    years_forward = days_forward / 365
    interpolated = _call_value(current_price, X, vol_surface(X), years_forward)
    first_derivative_discrete = np.gradient(interpolated, X)

    # to speed up TVR, we increase dx and therefore reduce n
    dx_2 = dx_1 * 10 # setting dx for the 2nd derivative to be 10x more sparse
    X_sparse = X[0::10] # array navigation: start at 0, go to end, every 10 values
    n_sparse = len(X_sparse)
    first_derivative_sparse = first_derivative_discrete[0::10] # start at 0, go to end, every 10 values

    # calculate second derivative of the call options prices using TVR
    diff_tvr = DiffTVR(n_sparse, dx_2)
    (y, _) = diff_tvr.get_deriv_tvr(
        data=first_derivative_sparse,
        deriv_guess=np.full(n_sparse + 1, 0.0),
        alpha=10,
        no_opt_steps=100,
    )    

    return (X_sparse, y[:len(X_sparse)])


def _call_value(S, K, sigma, t=0, r=0):
    # TODO: refactor this function (style)
    # use np.multiply and divide to handle divide-by-zero
    with np.errstate(divide="ignore"):
        d1 = np.divide(1, sigma * np.sqrt(t)) * (
            np.log(S / K) + (r + sigma ** 2 / 2) * t
        )
        d2 = d1 - sigma * np.sqrt(t)
    return np.multiply(norm.cdf(d1), S) - np.multiply(norm.cdf(d2), K * np.exp(-r * t))


def _call_vega(S, K, sigma, t=0, r=0):
    # TODO: refactor this function (style)
    with np.errstate(divide="ignore"):
        d1 = np.divide(1, sigma * np.sqrt(t)) * (
            np.log(S / K) + (r + sigma ** 2 / 2) * t
        )
    return np.multiply(S, norm.pdf(d1)) * np.sqrt(t)


def _bs_iv(
    price,
    S,
    K,
    t=0,
    r=0,
    precision=1e-4,
    initial_guess=0.2,
    max_iter=1000,
    verbose=False,
):
    # TODO: refactor this function (style)
    iv = initial_guess
    for _ in range(max_iter):
        P = _call_value(S, K, iv, t, r)
        diff = price - P
        if abs(diff) < precision:
            return iv
        grad = _call_vega(S, K, iv, t, r)
        iv += diff / grad
    if verbose:
        print(f"Did not converge after {max_iter} iterations")
    return iv
