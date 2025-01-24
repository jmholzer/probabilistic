from probabilistic import cli

input_csv_path = "data/SP_currentdate20240906_call20291221_currentprice550341.csv"
current_price = 5503.41
days_forward = 1933
output_csv_path = "/Users/henrytian/Downloads/results.csv"

df = cli.csv_runner.run(input_csv_path=input_csv_path, 
                   current_price=float(current_price), 
                   days_forward=int(days_forward), 
                   save_to_csv=False)