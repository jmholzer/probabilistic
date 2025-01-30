from oipd import cli
from datetime import datetime
import matplotlib.pyplot as plt

# example 1 - NVIDIA
input_csv_path = "data/nvidia_date20250128_strikedate20250516_price12144.csv"
current_price = 121.44
current_date = "2025-01-28"
strike_date = "2025-05-16"
# Convert the strings to datetime objects
current_date_dt = datetime.strptime(current_date, "%Y-%m-%d")
strike_date_dt = datetime.strptime(strike_date, "%Y-%m-%d")
# Calculate the difference in days
days_difference = (strike_date_dt - current_date_dt).days
# output_csv_path = "/Users/henrytian/Downloads/results.csv"

df = cli.generate_pdf.run(
    input_csv_path=input_csv_path,
    current_price=float(current_price),
    days_forward=int(days_difference),
    risk_free_rate=0.03,
    fit_kernel_pdf=True,
)

# Plot probability density function
plt.figure(figsize=(8, 5))
plt.plot(df.Price, df.PDF, label="Implied PDF", color="cyan", alpha=0.7)
# Add a vertical line at x = 121.44
plt.axvline(x=121.44, color="white", linestyle="--")
# Add annotation for clarity
plt.text(121.44, max(df.PDF) * 0.3, "Current price on\n Jan 28 2025: 121.44", 
         color="white", fontsize=12, ha="left", va="top")
# Labels and title
plt.xlabel("Price")
plt.ylabel("Density")
# plt.legend()
plt.title("Implied probability dist of NVIDIA on May 16 2025")
# Show the plot
plt.show()

# Example 2 - SPY
input_csv_path = "data/spy_date20250128_strike20250228_price60444.csv"
current_price = 604.44
current_date = "2025-01-28"
strike_date = "2025-02-28"
# Convert the strings to datetime objects
current_date_dt = datetime.strptime(current_date, "%Y-%m-%d")
strike_date_dt = datetime.strptime(strike_date, "%Y-%m-%d")
# Calculate the difference in days
days_difference = (strike_date_dt - current_date_dt).days
# output_csv_path = "/Users/henrytian/Downloads/results.csv"

df = cli.generate_pdf.run(
    input_csv_path=input_csv_path,
    current_price=float(current_price),
    days_forward=int(days_difference),
    risk_free_rate=0.03,
    fit_kernel_pdf=True,
    solver_method="brent",
)

# Plot probability density function
plt.figure(figsize=(8, 5))
plt.plot(df.Price, df.PDF, label="Implied PDF", color="cyan", alpha=0.7)
plt.xlabel("Price")
plt.ylabel("Density")
plt.legend()
plt.title("Implied PDF of S&P500 at 2025-02-28, from perspective of 2025-01-28")
plt.show()


# --- Example 3 - US Steel --- #
input_csv_path = "data/ussteel_date20250128_strike20251219_price3629.csv"
current_price = 36.29
current_date = "2025-01-28"
strike_date = "2025-12-19"
# Convert the strings to datetime objects
current_date_dt = datetime.strptime(current_date, "%Y-%m-%d")
strike_date_dt = datetime.strptime(strike_date, "%Y-%m-%d")
# Calculate the difference in days
days_difference = (strike_date_dt - current_date_dt).days
# output_csv_path = "/Users/henrytian/Downloads/results.csv"


ussteel_pdf = cli.generate_pdf.run(
    input_csv_path=input_csv_path,
    current_price=float(current_price),
    days_forward=int(days_difference),
    risk_free_rate=0.03,
    fit_kernel_pdf=True,
    solver_method="newton",
)

# Plot probability density function
plt.figure(figsize=(8, 5))
plt.plot(
    ussteel_pdf.Price, ussteel_pdf.PDF, label="Implied PDF", color="cyan", alpha=0.7
)
plt.xlabel("Price")
plt.ylabel("Density")
plt.legend()
plt.title(
    "Probability distribution of US Steel on 2025-12-19, from perspective of 2025-01-28"
)
plt.show()
