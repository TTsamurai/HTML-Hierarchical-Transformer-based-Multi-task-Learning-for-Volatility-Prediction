import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import ipdb

# Customized Transformers Util
print(os.getcwd())

from Sentence_Level_Transformers.custom_transformers.util import d, here, mask_
from Sentence_Level_Transformers.custom_transformers.transformers_gpu import *

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from sklearn.model_selection import train_test_split
from Sentence_Level_Transformers.custom_transformers import util

# from torchtext import data
from torch.utils.data import Dataset as VanillaDataset
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, math
from numpy.random import seed

# from tensorflow import set_random_seed
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import random, tqdm, sys, math, gzip

SEED_SPLIT = 42
GPT_FILE_TYPES = [
    "gpt_summary",
    "gpt_summary_overweight",
    "gpt_summary_underweight",
    "gpt_analysis_overweight",
    "gpt_analysis_underweight",
    "gpt_promotion_overweight",
    "gpt_promotion_underweight",
    "analysis_underweight_and_overweight",
    "summary_underweight_and_overweight",
    "analysis_and_summary_underweight_and_overweight",
]

VOL_SINGLE_PATH = [
    "train_split_SeriesSingleDayVol3.csv",
    "val_split_SeriesSingleDayVol3.csv",
    "test_split_SeriesSingleDayVol3.csv",
]
VOL_AVERAGE_PATH = [
    "train_split_Avg_Series_WITH_LOG.csv",
    "val_split_Avg_Series_WITH_LOG.csv",
    "test_split_Avg_Series_WITH_LOG.csv",
]
PRICE_PATH = [
    "train_price_label.csv",
    "dev_price_label.csv",
    "test_price_label.csv",
]


def get_text_embeddings_dict(file_path):
    text_embs = np.load(file_path, allow_pickle=True)
    return {key: text_embs[key] for key in text_embs}


def stack_text_embeddings(text_emb_dict, text_file_list):
    return np.stack([text_emb_dict[i] for i in text_file_list])


def get_text_path_label_prediction(text_file, labels, preds):
    return pd.DataFrame(
        {
            "text_file_name": text_file,
            "label": labels,
            "prediction": preds,
        }
    )


def load_and_concatenate_csv(file_names, base_path):
    dataframes = [
        pd.read_csv(os.path.join(base_path, file_name)) for file_name in file_names
    ]
    return pd.concat(dataframes, axis=0)


def calculate_stock_return(df, duration):
    """Calculates stock return and appends it as a new column."""
    return_col = f"stock_return_{duration}"
    future_col = f"future_{duration}"
    df[return_col] = (df[future_col] / df["current_adjclose_price"]) - 1
    return df


def create_datasets(features, labels, file_list, indices):
    """Extract subsets of data based on provided indices."""
    return features[indices], labels[indices], [file_list[i] for i in indices]


def split_data(indices, test_size, random_state):
    """Helper function to split data based on indices."""
    return train_test_split(indices, test_size=test_size, random_state=random_state)


def main_splitting(TEXT_emb, LABEL_emb, merged_text_file_list, SEED_SPLIT):
    all_indices = range(
        len(TEXT_emb)
    )  # Assuming all arrays are aligned and of the same length

    # Split indices into training + test and then train into train + validation
    train_val_indices, test_indices = split_data(
        all_indices, test_size=0.2, random_state=SEED_SPLIT
    )
    train_indices, val_indices = split_data(
        train_val_indices, test_size=0.125, random_state=SEED_SPLIT
    )

    # Use these indices to create datasets
    train_features, train_labels, train_file_list = create_datasets(
        TEXT_emb, LABEL_emb, merged_text_file_list, train_indices
    )
    val_features, val_labels, val_file_list = create_datasets(
        TEXT_emb, LABEL_emb, merged_text_file_list, val_indices
    )
    test_features, test_labels, test_file_list = create_datasets(
        TEXT_emb, LABEL_emb, merged_text_file_list, test_indices
    )

    return (
        (train_features, train_labels, train_file_list),
        (val_features, val_labels, val_file_list),
        (test_features, test_labels, test_file_list),
    )


def test_evaluate(arg, model, testloader):
    loss_test = 0.0
    for i, data in enumerate(testloader):
        inputs, labels, text_file = data
        inputs, labels = (
            torch.tensor(inputs, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32).cuda(),
        )
        if inputs.size(1) > arg.max_length:
            inputs = inputs[:, : arg.max_length, :]
        out_a = model(inputs)

        if (
            arg.task == "stock_price_prediction"
            or arg.task == "volatility_prediction"
            or arg.task == "stock_return_prediction"
        ):
            loss_function = nn.MSELoss()
        elif arg.task == "stock_movement_prediction":
            loss_function = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("task is not well defined")

        loss = loss_function(out_a, labels)
        loss_test += loss
        corresponding_text_file_label_predict = get_text_path_label_prediction(
            text_file, labels.cpu().detach().numpy(), out_a.cpu().detach().numpy()
        )
    acc = loss_test
    return acc, out_a, labels, corresponding_text_file_label_predict


def update_evaluation(
    evaluation,
    e,
    train_loss_tol,
    acc,
    out_a,
    labels,
    corresponding_text_file_label_predict,
):
    evaluation["epoch"].append(e)
    evaluation["Train Loss"].append(train_loss_tol.item())
    evaluation["Test Loss"].append(acc.item())
    evaluation["Outputs"].append(out_a.cpu().detach().numpy().tolist())
    evaluation["Actual"].append(labels.cpu().detach().numpy().tolist())
    evaluation["Text File Predict Label"].append(corresponding_text_file_label_predict)
    return evaluation


class Dataset_single_task(VanillaDataset):
    def __init__(self, texts, labels, text_file):
        "Initialization"
        self.labels = labels
        self.text = texts
        self.text_file = text_file

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.labels)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        if torch.is_tensor(index):
            index = index.tolist()

        # Load data and get label
        X = self.text[index, :, :]
        y = self.labels[index]
        text_file = self.text_file[index]
        return X, y, text_file


def go(arg):
    """
    Creates and trains a basic transformer for the volatility regression task.
    """
    LOG2E = math.log2(math.e)
    NUM_CLS = 1

    print(" Loading Data ...")
    # Text Embeddings Load
    text_file_path = os.path.join(arg.data_dir, arg.input_dir)
    TEXT_emb_dict = get_text_embeddings_dict(text_file_path)
    # Price Data Load
    price_data_path = os.path.join(arg.data_dir, arg.price_data_dir)

    # Load and concatenate CSV files
    vol_single_df = load_and_concatenate_csv(VOL_SINGLE_PATH, price_data_path)
    vol_average_df = load_and_concatenate_csv(VOL_AVERAGE_PATH, price_data_path)
    price_df = load_and_concatenate_csv(PRICE_PATH, price_data_path)

    # Filter and select only necessary columns
    text_file_list = list(TEXT_emb_dict.keys())
    vol_single_df = vol_single_df[vol_single_df["text_file_name"].isin(text_file_list)]
    vol_average_df = vol_average_df[
        vol_average_df["text_file_name"].isin(text_file_list)
    ]
    price_df = price_df[price_df["text_file_name"].isin(text_file_list)][
        [
            "text_file_name",
            f"future_{arg.duration}",
            f"future_label_{arg.duration}",
            "current_adjclose_price",
        ]
    ]
    vol_single_df = vol_single_df[["text_file_name", f"future_Single_{arg.duration}"]]
    vol_average_df = vol_average_df[["text_file_name", f"future_{arg.duration}"]]
    # Return Calculation
    price_df = calculate_stock_return(price_df, arg.duration)

    # Merging Text embeddigns and price data
    # タスクによりデータを変更する
    if arg.task == "stock_price_prediction":
        merged_data = pd.merge(
            vol_single_df, price_df, on="text_file_name", how="inner"
        )
        LABEL_emb = merged_data[f"future_{arg.duration}"].values
    elif arg.task == "stock_movement_prediction":
        merged_data = pd.merge(
            vol_single_df, price_df, on="text_file_name", how="inner"
        )
        LABEL_emb = merged_data[f"future_label_{arg.duration}"].values
    elif arg.task == "volatility_prediction":
        merged_data = pd.merge(
            vol_single_df, vol_average_df, on="text_file_name", how="inner"
        )
        LABEL_emb = merged_data[f"future_{arg.duration}"].values
    elif arg.task == "stock_return_prediction":
        merged_data = pd.merge(
            vol_single_df, price_df, on="text_file_name", how="inner"
        )
        LABEL_emb = merged_data[f"stock_return_{arg.duration}"].values
    else:
        raise ValueError("task is not well defined")

    # Prepare text emb
    merged_text_file_list = merged_data["text_file_name"].tolist()
    TEXT_emb = stack_text_embeddings(TEXT_emb_dict, merged_text_file_list)
    print(" Finish Loading Data... ")

    if arg.final:
        train, test = train_test_split(TEXT_emb, test_size=0.2, random_state=SEED_SPLIT)
        train_label, test_label = train_test_split(
            LABEL_emb, test_size=0.2, random_state=SEED_SPLIT
        )
        # train_label_b, test_label_b = train_test_split(LABEL_emb_b, test_size=0.2)

        training_set = Dataset_single_task(train, train_label)
        val_set = Dataset_single_task(test, test_label)
    else:
        train_data, val_data, _ = main_splitting(
            TEXT_emb, LABEL_emb, merged_text_file_list, SEED_SPLIT
        )
        train, train_label, text_file_train_label = train_data
        val, val_label, text_file_val_label = val_data

        training_set = Dataset_single_task(train, train_label, text_file_train_label)
        val_set = Dataset_single_task(val, val_label, text_file_val_label)

    if arg.train_normal_test_various:
        assert arg.file_name == "TextSequence", "TextSequence only"
        assert arg.final == False, "Not Final"
        various_test_set = {}
        for stance_file in GPT_FILE_TYPES:
            stance_input_dir = (
                f"./ptm_embeddings/{arg.embeddings_type}/{stance_file}.npz"
            )
            text_file_path = arg.data_dir + stance_input_dir
            TEXT_emb_dict = get_text_embeddings_dict(text_file_path)
            TEXT_emb_var = stack_text_embeddings(TEXT_emb_dict, merged_text_file_list)

            _, val_data_var, _ = main_splitting(
                TEXT_emb_var, LABEL_emb, merged_text_file_list, SEED_SPLIT
            )
            val_var, val_label_var, text_file_val_label_var = val_data_var
            assert text_file_val_label == text_file_val_label_var, "Text File Mismatch"
            various_dataset = Dataset_single_task(
                val_var, val_label_var, text_file_val_label_var
            )
            various_dataloader = torch.utils.data.DataLoader(
                various_dataset,
                batch_size=len(various_dataset),
                shuffle=False,
                num_workers=2,
            )
            various_test_set[stance_file] = various_dataloader

    trainloader = torch.utils.data.DataLoader(
        training_set, batch_size=arg.batch_size, shuffle=False, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        val_set, batch_size=len(val_set), shuffle=False, num_workers=2
    )
    print("training examples", len(training_set))

    if arg.final:
        print("test examples", len(val_set))
    else:
        print("validation examples", len(val_set))

    # create the model
    model = RTransformer_single_task(
        emb=arg.embedding_size,
        heads=arg.num_heads,
        depth=arg.depth,
        seq_length=arg.max_length,
        num_tokens=arg.vocab_size,
        num_classes=NUM_CLS,
        max_pool=arg.max_pool,
    )

    if arg.gpu:
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.cuda_id
            model.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())

    # training loop
    seen = 0
    evaluation = {
        "epoch": [],
        "Train Loss": [],
        "Test Loss": [],
        "Outputs": [],
        "Actual": [],
        "Text File Predict Label": [],
    }
    evaluation_various = {
        stance_file: {
            "epoch": [],
            "Train Loss": [],
            "Test Loss": [],
            "Outputs": [],
            "Actual": [],
            "Text File Predict Label": [],
        }
        for stance_file in GPT_FILE_TYPES
    }
    for e in tqdm.tqdm(range(arg.num_epochs)):
        train_loss_tol = 0.0
        print("\n epoch ", e)
        model.train(True)

        for i, data in enumerate(trainloader):
            # learning rate warmup
            # - we linearly increase the learning rate from 10e-10 to arg.lr over the first
            #   few thousand batches
            if arg.lr_warmup > 0 and seen < arg.lr_warmup:
                lr = max((arg.lr / arg.lr_warmup) * seen, 1e-10)
                opt.lr = lr

            opt.zero_grad()

            inputs, labels, _ = data
            inputs = Variable(inputs.type(torch.FloatTensor))
            labels = torch.tensor(labels, dtype=torch.float32).cuda()

            if inputs.size(1) > arg.max_length:
                inputs = inputs[:, : arg.max_length, :]

            out_a = model(inputs)
            # print(out_a.shape,out_b.shape)
            # print(out.shape,labels.shape)
            if (
                arg.task == "stock_price_prediction"
                or arg.task == "volatility_prediction"
                or arg.task == "stock_return_prediction"
            ):
                loss_function = nn.MSELoss()
            elif arg.task == "stock_movement_prediction":
                loss_function = nn.BCEWithLogitsLoss()
            else:
                raise ValueError("task is not well defined")
            loss = loss_function(out_a, labels)
            train_loss_tol += loss

            loss.backward()

            # clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if arg.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

            opt.step()

            seen += inputs.size(0)
        train_loss_tol = train_loss_tol / (i + 1)
        with torch.no_grad():

            model.train(False)
            acc, out_a, labels, corresponding_text_file_label_predict = test_evaluate(
                arg, model, testloader
            )
            if arg.train_normal_test_various:
                for stance_file in GPT_FILE_TYPES:
                    (
                        acc_var,
                        out_a_var,
                        labels_var,
                        corresponding_text_file_label_predict_var,
                    ) = test_evaluate(arg, model, various_test_set[stance_file])
                    evaluation_various[stance_file] = update_evaluation(
                        evaluation=evaluation_various[stance_file],
                        e=e,
                        train_loss_tol=train_loss_tol,
                        acc=acc_var,
                        out_a=out_a_var,
                        labels=labels_var,
                        corresponding_text_file_label_predict=corresponding_text_file_label_predict_var,
                    )
            evaluation = update_evaluation(
                evaluation=evaluation,
                e=e,
                train_loss_tol=train_loss_tol,
                acc=acc,
                out_a=out_a,
                labels=labels,
                corresponding_text_file_label_predict=corresponding_text_file_label_predict,
            )
    evaluation = pd.DataFrame(evaluation)
    evaluation.sort_values(["Test Loss"], ascending=True, inplace=True)
    if arg.train_normal_test_various:
        for stance_file in GPT_FILE_TYPES:
            evaluation_various[stance_file] = pd.DataFrame(
                evaluation_various[stance_file]
            )
            evaluation_various[stance_file].sort_values(
                ["Test Loss"], ascending=True, inplace=True
            )
        return evaluation, evaluation_various

    return evaluation
