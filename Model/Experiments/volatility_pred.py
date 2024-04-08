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
from concurrent.futures import ThreadPoolExecutor, as_completed


def set_seed(seed):
    """Set the seed for reproducibility."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # PyTorch
    if torch.cuda.is_available():  # If you're using PyTorch with a CUDA-capable device (GPU)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

FILE_TYPES = [
            "TextSequence",
            "ECT",
            "gpt_summary",
            "gpt_summary_overweight",
            "gpt_summary_underweight",
            "gpt_analysis_overweight",
            "gpt_analysis_underweight",
        ]
VOLATILITY_DURATIONS = [3,5,7,15,30]
def process_file_type(file_type):
    for vol_dur in VOLATILITY_DURATIONS:
        for seed in range(10):
            best_alpha = {"alpha": [], "best": []}
            for i in np.arange(0.1, 1.0, 0.1):
                parser = ArgumentParser()
                args = parser.parse_known_args()[0]
                args = easydict.EasyDict({
                    "num_epochs": 10,
                    "batch_size": 4,
                    "lr": 2e-5,
                    "tb_dir": "./runs",
                    "final": False,
                    "max_pool": False,
                    "embedding_size": 1024,
                    "max_length": 520,
                    "num_heads": 2,
                    "depth": 2,
                    "seed": 1,
                    "lr_warmup": 1000,
                    "gradient_clipping": 1.0,
                    "file_name": file_type,
                    "data_dir": "../../Data/",
                    "input_dir": f"./ptm_embeddings/{file_type}.npz",
                    "price_data_dir": "./price_data/",
                    "alpha": i,
                    "gpu": True,
                    "save": False,
                    "vol_duration": vol_dur,
                    "vocab_size": None,
                    "cuda_id": "0",
                    "volatility": True
                })

                evaluation = run_gpu.go(args)
                print("Results in alpha = ", i)
                print(evaluation)
                best_alpha["alpha"].append(i)
                best_alpha["best"].append(evaluation["Test Loss"].iloc[0])

            best_alpha = pd.DataFrame(best_alpha)
            best_alpha.sort_values(["best"], ascending=True, inplace=True)
            if args.volatility:
                save_dir = f"./results/volatility_prediction/{args.file_name}/{args.vol_duration}/"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                best_alpha.to_csv(save_dir + f"result_{seed}.csv")

# 時間かかりすぎるからfile typeで7並列
if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file_type, file_type): file_type for file_type in FILE_TYPES}
        for future in as_completed(futures):
            file_type = futures[future]
            try:
                data = future.result()
            except Exception as exc:
                print(f"{file_type} generated an exception: {exc}")
    # for file_type in FILE_TYPES: # 7
    #     for vol_dur in VOLATILITY_DURATIONS: # 5
    #         for seed in range(10): # 10
    #             best_alpha = {"alpha": [], "best": []} # 10
    #             # for i in np.arange(0.1, 0.2, 0.1):
    #             for i in np.arange(0.1, 1.0, 0.1): # 10 * 10
    #                 # print('OPTIONS ', options)
    #                 # Tuning Parameters:
    #                 parser = ArgumentParser()
    #                 args = parser.parse_known_args()[0]
    #                 args = easydict.EasyDict(
    #                     {
    #                         "num_epochs": 10,
    #                         "batch_size": 4,
    #                         "lr": 2e-5,
    #                         "tb_dir": "./runs",
    #                         "final": False,
    #                         "max_pool": False,
    #                         "embedding_size": 1024,  # 1024(textual feature)
    #                         "max_length": 520,
    #                         "num_heads": 2,
    #                         "depth": 2,
    #                         "seed": 1,
    #                         "lr_warmup": 1000,
    #                         "gradient_clipping": 1.0,
    #                         "file_name": file_type,
    #                         "data_dir": "../../Data/",
    #                         "input_dir": f"./ptm_embeddings/{file_type}.npz",
    #                         "price_data_dir": "./price_data/",
    #                         "alpha": i,
    #                         "gpu": True,
    #                         "save": False,
    #                         "vol_duration": vol_dur,
    #                         "vocab_size": None,
    #                         "cuda_id": "0",
    #                         "volatility": True
    #                     }
    #                 )

    #                 evaluation = run_gpu.go(args)
    #                 # import ipdb;ipdb.set_trace()
    #                 print("Results in alpha = ", i)
    #                 print(evaluation)
    #                 best_alpha["alpha"].append(i)
    #                 best_alpha["best"].append(evaluation["Test Loss"].iloc[0])

    #             best_alpha = pd.DataFrame(best_alpha)
    #             best_alpha.sort_values(["best"], ascending=True, inplace=True)
    #             if args.volatility:
    #                 save_dir = f"./results/volatility_prediction/{args.file_name}/{args.vol_duration}/"
    #                 if not os.path.exists(save_dir):
    #                     os.makedirs(save_dir)
    #                 best_alpha.to_csv(save_dir + f"result_{seed}.csv")
