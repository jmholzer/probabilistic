![Probabilistic logo](probabilistic/dashboard/resources/logo.png)

![Python version](https://img.shields.io/badge/python-3.10-blue.svg)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black)

This Python project generates probability density function (PDFs) and cumulative distribution functions (CDFs) for the future prices of stocks, as implied by options prices. The output is visualized with matplotlib, and the project also includes a user-friendly web-based dashboard interface built with Streamlit.


## Table of Contents

- [Installation](#installation)
- [Quick Start Guide](#Quick-Start-Guide)
- [Algorithm Overview](#algorithm-overview)
- [License](#license)


## Installation

```pip install probabilistic```

Please note that this project requires Python 3.10 or later.


## Quick Start Guide

<b>Option 1: To use probabilistic, see `example_script.py` for a demo:</b>

The user will need to specify 4 mandatory arguments:

1. `input_csv_path`: a string containing the file path of the options data in a csv, with the columns 'strike', 'last_price', 'bid, 'ask'
2. `current_price`: a number of the underlying asset's current price
3. `days_foward`: a number of the days between the current date and the strike date
4. `risk_free_rate`: a number indicating the annual risk-free rate in nominal terms

There are 4 additional optional arguments:

5. `fit_kernel_pdf`: (optional) a True or False boolean, indicating whether to fit a kernel-density estimator on the resulting raw probability distribution. Fitting a KDE may improve edge-behavior of the PDF. Default is False
6. `save_to_csv`: (optional) a True or False boolean, where if True, the output will be saved to csv. Default is False
7. `output_csv_path`: (optional) a string containing the file path where the user wishes to save the results
8. `solver_method`: (optional) a string of either 'newton' or 'brent', indicating which solver to use. Default is 'brent'

3 examples of options data is provided in the `data/` folder. 

```
from probabilistic import cli

input_csv_path = "path_to_your_options_data"
current_price = 184.8
days_forward = 123

df = cli.csv_runner.run(
    input_csv_path=input_csv_path,
    current_price=float(current_price),
    days_forward=int(days_difference),
    risk_free_rate=0.03,
    fit_kernel_pdf=True,
)
```

![Probabilistic example output](.meta/images/nvidia_output.png)


<b>Option 2: To start the web-based dashboard, run the following command:</b>

```bash
probabilistic
```

This will start a local web server and you should be able to access the dashboard in your web browser at `localhost:8501`.

The user will need to provide their own options data in a CSV file with the columns 'strike', and 'last_price'. Sample data for SPY can be found in the `data` folder.


## Theory Overview

An option is a financial derivative that gives the holder the right, but not the obligation, to buy or sell an asset at a specified price (strike price) on a certain date in the future. Intuitively, the value of an option depends on the probability that it will be profitable or "in-the-money" at expiration. If the probability of ending "in-the-money" (ITM) is high, the option is more valuable. If the probability is low, the option is worth less.

As an example, imagine Apple stock (AAPL) is currently $150, and you buy a call option with a strike price of $160 (meaning you can buy Apple at $160 at expiration).
- If Apple is likely to rise to $170, the option has a high probability of being ITM → more valuable
- If Apple is unlikely to go above $160, the option has little chance of being ITM → less valuable

This illustrates how option prices contain information about the probabilities about the future price of the underlying stock (as determined by market expectations). By knowing the prices of options, we can reverse-engineer and extract this information about the probabilities. 

For a simplified worked example, see this [excellent blog post](https://reasonabledeviations.com/2020/10/01/option-implied-pdfs/).
For a complete reading of the financial theory, see [this paper](https://www.bankofengland.co.uk/-/media/boe/files/quarterly-bulletin/2000/recent-developments-in-extracting-information-from-options-markets.pdf?la=en&hash=8D29F2572E08B9F2B541C04102DE181C791DB870).


## Algorithm Overview

The process of generating the PDFs and CDFs is as follows:

1. For an underlying asset, options data along the full range of strike prices are read from a CSV file to create a DataFrame. This gives us a table of strike prices along with the last price[^1] each option sold for
2. Using the Black-Sholes formula, we convert strike prices into implied volatilities (IV)[^2]. IV are solved using either Newton's Method or Brent's root-finding algorithm, as specified by the `solver_method` argument. 
3. Using B-spline, we fit a curve-of-best-fit onto the resulting IVs over the full range of strike prices[^3]. Thus, we have extracted a continuous model from discrete IV observations - this is called the volatility smile
4. From the volatility smile, we use Black-Scholes to convert IVs back to prices. Thus, we arrive at a continuous curve of options prices along the full range of strike prices
5. From the continuous price curve, we use numerical differentiation to get the first derivative of prices. Then we numerically differentiate again to get the second derivative of prices. The second derivative of prices multiplied by a discount factor $\exp^{r*\uptau}$, results in the probability density function [^4]
6. We can fit a KDE onto the resulting PDF, which in some cases will improve edge-behavior at very high or very low prices. This is specified by the argument `fit_kernal_pdf`
7. Once we have the PDF, we can calculate the CDF


[^1]: We chose to use last price instead of calculating the mid-price given the bid-ask spread. This is because Yahoo Finance, a common source for options chain data, often lacks bid-ask data. See for example [Apple options](https://finance.yahoo.com/quote/AAPL/options/)
[^2]: We convert from price-space to IV-space, and then back to price-space as described in step 4. See this [blog post](https://reasonabledeviations.com/2020/10/10/option-implied-pdfs-2/) for a breakdown of why we do this double conversion
[^3]: See [this paper](https://edoc.hu-berlin.de/bitstream/handle/18452/14708/zeng.pdf?sequence=1&isAllowed=y) for more details. In summary, options markets contains noise. Therefore, generating a volatility smile through simple interpolation will result in a noisy smile function. Then converting back to price-space will result in a noisy price curve. And finally when we numerically twice differentiate the price curve, noise will be amplified and the resulting PDF will be meaningless. Thus, we need either a parametric or non-parametric model to try to extract the true relationship between IV and strike price from the noisy observations. The paper suggests a 3rd order B-spline as a possible model choice
[^4]: For a proof of this derivation, see this [blog post](https://reasonabledeviations.com/2020/10/10/option-implied-pdfs-2/)


## License

This project is a preview, it is not currently licensed. Not financial advice.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
