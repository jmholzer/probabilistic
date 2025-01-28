from probabilistic import cli
from datetime import datetime
import matplotlib.pyplot as plt

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

df = cli.csv_runner.run(
    input_csv_path=input_csv_path,
    current_price=float(current_price),
    days_forward=int(days_difference),
    fit_kernel_pdf=False,
)

# Plot probability density function
plt.figure(figsize=(8, 5))
plt.plot(df.Price, df.PDF, label="Implied PDF", color="cyan", alpha=0.7)
plt.xlabel("Price")
plt.ylabel("Density")
plt.legend()
plt.title("Implied PDF of NVIDIA on ")
plt.show()
