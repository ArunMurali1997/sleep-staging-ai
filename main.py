import os
import argparse

from train_models import train_pipeline, get_data_loaders, preprocess
from distillation_train import train_distillation
from config.config import  META_FILE

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["train","distill"],
        required=True
    )

    args = parser.parse_args()

    if args.mode == "train":
        print("Running normal training (CNN + ViT)")
        train_pipeline()

    elif args.mode == "distill":
        if not os.path.exists(META_FILE):
            print("Starting preprocessing")
            preprocess()
        print("Running knowledge distillation")
        train_loader, test_loader, class_weights = get_data_loaders()
        train_distillation(train_loader, test_loader, class_weights)

if __name__ == "__main__":
    main()