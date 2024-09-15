from probabilistic import cli

input_csv_path = "data/AAPL_currentdateNov14_callMar15_currentprice18480_CLEAN.csv"
current_price = 184.8
days_forward = 123
output_csv_path = "/Users/henrytian/Downloads/results.csv"

cli.csv_runner.run(input_csv_path, float(current_price), int(days_forward), output_csv_path)