import sys
import os

sys.path.append("../../Model")
import torch
import random 
import transformers
from Sentence_Level_Transformers import run_gpu
import numpy as np
import pandas as pd
import easydict
from argparse import ArgumentParser


def set_seed(seed):
    """Set the seed for reproducibility."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # PyTorch
    if torch.cuda.is_available():  # If you're using PyTorch with a CUDA-capable device (GPU)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

if __name__ == "__main__":
    for seed in range(10):
        best_alpha = {"alpha": [], "best": []}
        # for i in np.arange(0.1, 0.2, 0.1):
        for i in np.arange(0.1, 1.0, 0.1):
            # print('OPTIONS ', options)
            # Tuning Parameters:
            parser = ArgumentParser()
            args = parser.parse_known_args()[0]
            args = easydict.EasyDict(
                {
                    "num_epochs": 10,
                    "batch_size": 4,
                    "lr": 2e-5,
                    "tb_dir": "./runs",
                    "final": False,
                    "max_pool": False,
                    "embedding_size": 1024,  # 1024(textual feature)
                    "max_length": 520,
                    "num_heads": 2,
                    "depth": 2,
                    "seed": 1,
                    "lr_warmup": 1000,
                    "gradient_clipping": 1.0,
                    "file_name": "TextSequence",
                    "data_dir": "../../Data/",
                    "input_dir": "./ptm_embeddings/TextSequence.npz",
                    "price_data_dir": "./price_data/",
                    "label_dir": "../../Price_Data/volitility_following_3days.npy",
                    "label_dir_b": "../../Price_Data/volitility_single_3days.npy",
                    "alpha": i,
                    "gpu": True,
                    "save": False,
                    "vol_duration": 3,
                    "vocab_size": None,
                    "cuda_id": "0",
                    "volatility": True
                }
            )

            evaluation = run_gpu.go(args)
            # import ipdb;ipdb.set_trace()
            print("Results in alpha = ", i)
            print(evaluation)
            best_alpha["alpha"].append(i)
            best_alpha["best"].append(evaluation["Test Loss"].iloc[0])

        best_alpha = pd.DataFrame(best_alpha)
        best_alpha.sort_values(["best"], ascending=True, inplace=True)
        if args.volatility:
            best_alpha.to_csv(f"./results/volatility_prediction/{args.vol_duration}_days_result_{seed}.csv")

        best_alpha
