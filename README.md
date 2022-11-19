# probabilistic
Generate future price PDFs for publicly traded securities using options data

# Installation Steps

1. `git clone https://github.com/jmholzer/probabilistic`
2. `python3.10 -m venv venv`
3. `source venv/bin/activate`
4. `pip install -r requirements.txt`
5. `pip install -e .`

# Test Command

`probabilistic calculate --csv tests/io/resources/sample.csv --current-price 377 --days-forward 21`
