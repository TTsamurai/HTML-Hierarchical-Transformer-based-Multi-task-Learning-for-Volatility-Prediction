import sys
import os

sys.path.append("../../Model")
import torch
import random
import transformers
from Sentence_Level_Transformers import run_gpu_single_task
import numpy as np
import pandas as pd
import easydict
from argparse import ArgumentParser


def set_seed(seed):
    """Set the seed for reproducibility."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # PyTorch
    if (
        torch.cuda.is_available()
    ):  # If you're using PyTorch with a CUDA-capable device (GPU)
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
DURATIONS = [3, 7, 15, 30]
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="stock_movement_prediction",
        choices=[
            "stock_price_prediction",
            "stock_movement_prediction",
            "volatility_prediction",
            "stock_return_prediction",
        ],
    )
    args = parser.parse_known_args()[0]
    # Convert args to a mutable EasyDict
    args = easydict.EasyDict(vars(args))
    for file_type in FILE_TYPES:
        for dur in DURATIONS:
            best_alpha = {"best": [], "seed": [], "prediction": [], "actual": []}
            for seed in range(10):
                args.update(
                    {
                        "num_epochs": 10,
                        "batch_size": 16,
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
                        "file_name": file_type,
                        "data_dir": "../../Data/",
                        "input_dir": f"./ptm_embeddings/{file_type}.npz",
                        "price_data_dir": "./price_data/",
                        # "alpha": i,
                        "gpu": True,
                        "save": False,
                        "duration": dur,
                        "vocab_size": None,
                        "cuda_id": "0",
                    }
                )

                evaluation = run_gpu_single_task.go(easydict.EasyDict(args))
                print(evaluation)
                best_alpha["best"].append(evaluation["Test Loss"].iloc[0])
                best_alpha["seed"].append(seed)
                best_alpha["prediction"].append(evaluation["Outputs"].iloc[0])
                best_alpha["actual"].append(evaluation["Actual"].iloc[0])

            best_alpha = pd.DataFrame(best_alpha)
            best_alpha.sort_values(["best"], ascending=True, inplace=True)
            save_dir = f"./results/{args.task}/{args.file_name}/{args.duration}/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            best_alpha.to_csv(save_dir + f"result.csv")
