# Overview
Package generates the forward-looking distribution (probability density function, hereafter referred to as 'pdf') of an asset's price at a particular date in the future.

For example, the below graphic shows the pdf of SPY price on Oct 31 2022, generated only from information that existed on Oct 06 2022 (SPY was $373.2 on Oct 06).

![pdf of SPY price on 2022-10-31, on 2022-10-06](https://github.com/jmholzer/probabilistic/blob/main/TVR%202nd%20derivative%20alpha10.png)

# Simplified theory
Source: https://www.bankofengland.co.uk/-/media/boe/files/quarterly-bulletin/2000/recent-developments-in-extracting-information-from-options-markets.pdf?la=en&hash=8D29F2572E08B9F2B541C04102DE181C791DB870

Options are contracts used to insure against or speculate/take a view on uncertainty about the future prices of a wide range of financial assets and physical commodities. The prices at which options are traded contain information about the markets’ uncertainty about the future prices of these ‘underlying’ assets. 

Said differently, an option's price is a function of the probability of its event occuring. Therefore, we can use a range of options prices to reverse-engineer the implied pdf of an asset's price in the future. This implied pdf can be considered the market's consensus on what the future price will be.

If we think about the asset's price at some date in the future as a random variable (RV), this RV will have a true pdf. This true pdf can never be observed, but there's certain methods we can use to make our best guess of it. 
1. assume the best guess for the RV's true pdf is its pdf based on historical data
2. assume the best guess for the RV's true pdf is the market's consensus of its future pdf (which is what this package finds)
3. use private information to form an expectation about the RV's true pdf

Method 1 is flawed as an asset's historical price distribution is not necessarily indicative of its future distribution. This can be easily seen from looking at S&P's 1 month forward-looking distribution at Oct 1 2022, when the risk of nuclear war was higher than ever before. Compared to historical price distribution, the 1-month forward-looking pdf in Oct had a higher left tail (the market had priced in a 0.46% probability that the S&P falls to $2600 by end of month). If you used the historical distribution, left-tail risk would have been underestimated. 

This package can be used to make positive expected value (EV) bets if you can utilize method 3. If you have some hypothesis that differs from the market consensus, and you use this package to find that the market is under-predicting or over-predicting some realization, then there are positive EV trades you can perform. For example, if you believed that the risk of nuclear war within 30 days on Oct 1 was 10%, and you believed that nuclear war would cause S&P price to fall below $2600, then the price of a put on S&P with a strike price of $2600 would be undervalued, and purchasing it would be a positive EV bet. 

# Why use this package?
There seems to no existing package to generate future-looking pdfs of an asset's price. If a user wanted to do this themselves, they'd have to wade through academic finance papers to learn the theory and then attempt to put it into code. We provide an off-the-shelf solution. 

# Installation Steps

1. `git clone https://github.com/jmholzer/probabilistic`
2. `cd probabilistic`
3. `python3.10 -m venv venv`
4. `source venv/bin/activate` or `venv\Scripts\Activate` on Windows
5. `pip install -r requirements.txt`
6. `pip install -e .`

# Test Command

`probabilistic calculate --csv tests/io/resources/sample.csv --current-price 377 --days-forward 21`
