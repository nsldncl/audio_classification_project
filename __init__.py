from transformer_1 import train
from create_dataset import create_dataset


if __name__ == "__main__":
    print("Start")
    x = input("Create dataset for yes-Y for no-N")
    if x == "Y":
        create_dataset()
        start_train() 