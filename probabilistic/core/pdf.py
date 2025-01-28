from typing import Dict, Tuple

import numpy as np
from pandas import concat, DataFrame
from scipy.integrate import simps
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import norm, gaussian_kde


def fit_kde(pdf_point_arrays: tuple) -> tuple:
    """
    Fits a Kernel Density Estimation (KDE) to the given implied probability density function (PDF).

    Args:
        pdf_point_arrays (tuple): A tuple containing:
            - A numpy array of price values
            - A numpy array of PDF values

    Returns:
        tuple: (prices, fitted_pdf), where:
            - prices: The original price array
            - fitted_pdf: The KDE-fitted probability density values
    """

    # Unpack tuple
    prices, pdf_values = pdf_point_arrays

    # Normalize PDF to ensure it integrates to 1
    pdf_values /= np.trapz(pdf_values, prices)  # Use trapezoidal rule for normalization

    # Fit KDE using price points weighted by the normalized PDF
    kde = gaussian_kde(prices, weights=pdf_values)

    # Generate KDE-fitted PDF values
    fitted_pdf = kde.pdf(prices)

    return (prices, fitted_pdf)


def calculate_pdf(
    options_data: DataFrame, current_price: float, days_forward: int
) -> Tuple[np.ndarray]:
    """The main execution path for the pdf module. Takes a `DataFrame` of
    options data as input and makes a series of function calls to

    Args:
        options_data: a DataFrame containing options price data with
            cols ['strike', 'last_price']
        current_price: the current price of the security
        days_forward: the number of days in the future to estimate the
            price probability density at

    Returns:
        a tuple containing the price and density values (in numpy arrays)
        of the calculated PDF
    """
    options_data, min_strike, max_strike = _extrapolate_call_prices(
        options_data, current_price
    )
    options_data = _calculate_last_price(options_data)
    options_data = _calculate_IV(options_data, current_price, days_forward)
    denoised_iv = _fit_bspline_IV(options_data)
    pdf = _create_pdf_point_arrays(denoised_iv, current_price, days_forward)
    return _crop_pdf(pdf, min_strike, max_strike)


def calculate_cdf(
    pdf_point_arrays: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the cumulative probability at each price. Takes as input the array
    of pdf and array of prices, and calculates the cumulative probability as the
    numerical integral over the pdf function.

    For simplicity, it assumes that the CDF at the starting price
        = 1 - 0.5*(total area of the pdf)
    and therefore it adds 0.5*(total area of the pdf) to every cdf for the
    remainder of the domain

    Args:
        pdf_point_arrays: a tuple containing arrays representing a PDF

    Returns:
        A tuple containing the price domain and the point values of the CDF
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
            integral = (
                simps(y=pdf_array[i - 1 : i + 1], x=x_array[i - 1 : i + 1]) + cdf[-1]
            )
        cdf.append(integral)

    return (x_array, cdf)


def calculate_quartiles(
    cdf_point_arrays: Tuple[np.ndarray, np.ndarray],
) -> Dict[float, float]:
    """

    Args:
        cdf_point_arrays: a tuple containing arrays representing a CDF

    Returns:
        a DataFrame containing the quartiles of the given CDF
    """
    cdf_interpolated = interp1d(cdf_point_arrays[0], cdf_point_arrays[1])
    x_start, x_end = cdf_point_arrays[0][0], cdf_point_arrays[0][-1]
    return {
        0.25: brentq(lambda x: cdf_interpolated(x) - 0.25, x_start, x_end),
        0.5: brentq(lambda x: cdf_interpolated(x) - 0.5, x_start, x_end),
        0.75: brentq(lambda x: cdf_interpolated(x) - 0.75, x_start, x_end),
    }


def _extrapolate_call_prices(
    options_data: DataFrame, current_price: float
) -> tuple[DataFrame, int, int]:
    """Extrapolate the price of the call options to strike prices outside
    the range of options_data. Extrapolation is done to zero and twice the
    highest strike price in options_data. Done to give the resulting PDF
    more stability.

    Args:
        options_data: a DataFrame containing options price data with
            cols ['strike', 'last_price']
        current_price: the current price of the security

    Returns:
        the extended options_data DataFrame
    """
    min_strike = int(options_data.strike.min())
    max_strike = int(options_data.strike.max())
    lower_extrapolation = DataFrame(
        {"strike": p, "last_price": current_price - p} for p in range(0, min_strike)
    )
    upper_extrapolation = DataFrame(
        {
            "strike": p,
            "last_price": 0,
        }
        for p in range(max_strike + 1, max_strike * 2)
    )
    return (
        concat([lower_extrapolation, options_data, upper_extrapolation]),
        min_strike,
        max_strike,
    )


def _calculate_last_price(options_data: DataFrame) -> DataFrame:
    """Take the last-price of the options at each strike price.

    Args:
        options_data: a DataFrame containing options price data with
            cols ['strike', 'last_price']

    Returns:
        the options_data DataFrame, with an additional column for mid-price
    """
    options_data["last_price"] = options_data["last_price"]
    options_data = options_data[options_data.last_price >= 0]
    return options_data


def _calculate_IV(
    options_data: DataFrame, current_price: float, days_forward: int
) -> DataFrame:
    """Calculate the implied volatility of the options in options_data

    Args:
        options_data: a DataFrame containing options price data with
            cols ['strike', 'last_price']
        current_price: the current price of the security
        days_forward: the number of days in the future to estimate the
            price probability density at

    Returns:
        the options_data DataFrame, with an additional column for implied volatility
    """
    years_forward = days_forward / 365
    options_data["iv"] = options_data.apply(
        lambda row: _bs_iv(
            row.last_price,
            current_price,
            row.strike,
            years_forward,
        ),
        axis=1,
    )
    options_data = options_data.dropna()
    return options_data


def _fit_bspline_IV(options_data: DataFrame) -> DataFrame:
    """Fit a bspline function on the IV observations, in effect denoising the IV.
        From this smoothed IV function, generate (x,y) coordinates
        representing observations of the denoised IV

    Args:
        options_data: a DataFrame containing options price data with
            cols ['strike', 'last_price', 'iv']

    Returns:
        a tuple containing x-axis values (index 0) and y-axis values (index 1)
        'x' represents the price
        'y' represents the value of the IV
    """
    x = options_data["strike"]
    y = options_data["iv"]

    # fit the bspline using scipy.interpolate.splrep, with k=3
    """
    Bspline Parameters:
        t = the vector of knots
        c = the B-spline coefficients
        k = the degree of the spline
    """
    tck = interpolate.splrep(x, y, s=10, k=3)

    dx = 0.1  # setting dx = 0.1 for numerical differentiation
    domain = int((max(x) - min(x)) / dx)

    # compute (x,y) observations of the denoised IV from the fitted IV function
    x_new = np.linspace(min(x), max(x), domain)
    y_fit = interpolate.BSpline(*tck)(x_new)

    return (x_new, y_fit)


def _create_pdf_point_arrays(
    denoised_iv: tuple, current_price: float, days_forward: int, rf=0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """Create two arrays containing x- and y-axis values representing a calculated
    price PDF

    Args:
        denoised_iv: (x,y) observations of the denoised IV
        current_price: the current price of the security
        days_forward: the number of days in the future to estimate the
            price probability density at
        rf: the current risk free interest rate

    Returns:
        a tuple containing x-axis values (index 0) and y-axis values (index 1)
    """

    # extract the x and y vectors from the denoised IV observations
    x_IV = denoised_iv[0]
    y_IV = denoised_iv[1]

    # convert IV-space to price-space
    # re-values call options using the BS formula, taking in as inputs S, domain, IV, and time to expiry
    years_forward = days_forward / 365
    interpolated = _call_value(current_price, x_IV, y_IV, years_forward)
    first_derivative_discrete = np.gradient(interpolated, x_IV)
    second_derivative_discrete = np.gradient(first_derivative_discrete, x_IV)

    # apply coefficient to reflect the time value of money
    pdf = np.exp(rf * years_forward) * second_derivative_discrete

    return (x_IV, pdf)


def _crop_pdf(
    pdf: Tuple[np.ndarray, np.ndarray], min_strike: float, max_strike: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop the PDF to the range of the original options data"""
    l, r = 0, len(pdf[0]) - 1
    while pdf[0][l] < min_strike:
        l += 1
    while pdf[0][r] > max_strike:
        r -= 1
    return pdf[0][l : r + 1], pdf[1][l : r + 1]


# def _bs_iv(
#     price,
#     S,
#     K,
#     t=0,
#     r=0,
#     precision=1e-4,
#     initial_guess=0.2,
#     max_iter=1000,
#     verbose=False,
# ):
#     # TODO: refactor this function (style)
#     iv = initial_guess
#     for _ in range(max_iter):
#         P = _call_value(S, K, iv, t, r)
#         diff = price - P
#         if abs(diff) < precision:
#             return iv
#         grad = _call_vega(S, K, iv, t, r)
#         iv += diff / grad
#     if verbose:
#         print(f"Did not converge after {max_iter} iterations")
#     return iv


def _bs_iv(price, S, K, t, r=0):
    """
    Computes the implied volatility (IV) of a European call option using Brent’s method.

    This function finds the implied volatility by solving for sigma (volatility) in the
    Black-Scholes pricing formula. It uses Brent’s root-finding algorithm to find the
    volatility that equates the Black-Scholes model price to the observed market price.

    Args:
        price (float): The observed market price of the option.
        S (float): The current price of the underlying asset.
        K (float): The strike price of the option.
        t (float): Time to expiration in years.
        r (float, optional): The risk-free interest rate (annualized). Defaults to 0.

    Returns:
        float: The implied volatility (IV) if a solution is found.
        np.nan: If the function fails to converge to a solution.

    Raises:
        ValueError: If Brent’s method fails to find a root in the given range.

    Notes:
        - The function searches for IV within the range [1e-6, 5.0] (0.0001% to 500% volatility).
        - If `t <= 0`, the function returns NaN since volatility is undefined for expired options.
        - If the function fails to converge, it returns NaN instead of raising an exception.
    """

    if t <= 0:
        return np.nan  # No volatility if time is zero or negative

    try:
        return brentq(lambda iv: _call_value(S, K, iv, t, r) - price, 1e-6, 5.0)
    except ValueError:
        return np.nan  # Return NaN if no solution is found


def _call_value(S, K, sigma, t=0, r=0):
    # TODO: refactor this function (style)
    # use np.multiply and divide to handle divide-by-zero
    with np.errstate(divide="ignore"):
        d1 = np.divide(1, sigma * np.sqrt(t)) * (np.log(S / K) + (r + sigma**2 / 2) * t)
        d2 = d1 - sigma * np.sqrt(t)
    return np.multiply(norm.cdf(d1), S) - np.multiply(norm.cdf(d2), K * np.exp(-r * t))


def _call_vega(S, K, sigma, t=0, r=0):
    # TODO: refactor this function (style)
    with np.errstate(divide="ignore"):
        d1 = np.divide(1, sigma * np.sqrt(t)) * (np.log(S / K) + (r + sigma**2 / 2) * t)
    return np.multiply(S, norm.pdf(d1)) * np.sqrt(t)
