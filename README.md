# Probabilistic

This Python project generates future price probability density functions (PDFs), cumulative distribution functions (CDFs), and quartiles for publicly traded securities using options data. The output is visualized with matplotlib, and the project also includes a user-friendly web-based dashboard interface built with Streamlit.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Overview](#algorithm-overview)
- [License](#license)
- [Contact](#contact)

## Installation

1. Clone the repo

```bash
git clone https://github.com/jmholzer/probabilistic-pdfs.git
```

2. Navigate to the project directory

```bash
cd probabilistic-pdfs
```

3. Install Python dependencies

```bash
pip install -r requirements.txt
```

4. Install the project

```bash
pip install .
```

Please note that this project requires Python 3.10 or later.

## Usage

To start the web-based dashboard, run the following command:

```bash
probabilistic
```

This will start a local web server and you should be able to access the dashboard in your web browser at `localhost:8501`.

The user will need to provide their own options data in a CSV file with the headers 'strike', 'bid', and 'ask'. Sample data for SPY can be found in the `data` folder.

## Algorithm Overview

The process of generating the PDFs, CDFs, and quartiles is as follows:

1. Options data is read from a CSV file to create a DataFrame.
2. The mid-price of each option is calculated, which is simply the average of the bid and ask prices.
3. The implied volatility (IV) of each option is then computed using the Black-Scholes formula. The IV is a measure of how much the market expects the price of the asset to move in the future.
4. Two arrays of x-values (prices) and y-values (densities) are produced, representing the PDF of the future price of the asset.
5. The cumulative probability at each price is calculated, resulting in the CDF.
6. Quartiles (25th, 50th, and 75th percentiles) of the price distribution are derived.

This tool can provide insights into market expectations for the future price of an asset based on current options prices. For instance, if you want to anticipate the likely price of a stock 30 days from now, you could use this tool to calculate the PDF and CDF from the stock's current options data, and inspect the distribution quartiles to understand the range of probable prices.

## License

This project is a preview, it is not currently licensed. Not financial advice.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
