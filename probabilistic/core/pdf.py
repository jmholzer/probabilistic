from typing import Tuple

import numpy as np
from pandas import DataFrame
from scipy.interpolate import interp1d
from scipy.stats import norm

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

    X = np.arange(options_data.strike.min(), options_data.strike.max(), 0.05)

    # re-values call options using the BS formula, taking in as inputs S, domain, IV, and time to expiry
    interpolated = _call_value(current_price, X, vol_surface(X), days_forward)
    first_derivative_discrete = np.gradient(interpolated, X)

    # calculate second derivative of the call options prices using TVR
    diff_tvr = DiffTVR(len(interpolated), 0.05)
    (y, _) = diff_tvr.get_deriv_tvr(
        data=first_derivative_discrete,
        deriv_guess=np.full(len(interpolated) + 1, 0.0),
        alpha=10,
        no_opt_steps=100,
    )

    return (X, y)


def _call_value(S, K, sigma, t=0, r=0):
    # use np.multiply and divide to handle divide-by-zero
    with np.errstate(divide="ignore"):
        d1 = np.divide(1, sigma * np.sqrt(t)) * (
            np.log(S / K) + (r + sigma ** 2 / 2) * t
        )
        d2 = d1 - sigma * np.sqrt(t)
    return np.multiply(norm.cdf(d1), S) - np.multiply(norm.cdf(d2), K * np.exp(-r * t))


def _call_vega(S, K, sigma, t=0, r=0):
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
