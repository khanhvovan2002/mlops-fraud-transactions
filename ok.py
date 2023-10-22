import neptune


run = neptune.init_run(
    project="khanhvovan2002/fraud-transaction-",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YjdmYzg5MS1lODg1LTRlODktYWYzZS02NmI1YzdiYjU0MjIifQ==",
)  # y
run["train/dataset"].track_files("adult.csv")

run.stop()