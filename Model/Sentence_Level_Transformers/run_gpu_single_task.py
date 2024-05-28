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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.append("../Experiments")
from single_task_prediction import GPT_FILE_TYPES

SEED_SPLIT = 42

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
    return {
        text_file[i]: {"label": labels[i], "pred": preds[i]}
        for i in range(len(text_file))
    }


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


def get_acl_dataloader(
    arg, stance_file, merged_text_file_list, LABEL_emb, text_file_val_label
):
    stance_input_dir = f"./ptm_embeddings/{arg.embeddings_type}/{stance_file}.npz"
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
    return various_dataloader


def get_ectsum_dataloader(arg, stance_file):
    # text_emb, label_emb, text_file_listを出す
    stance_input_dir = f"ptm_embeddings/{arg.embeddings_type}/ectsum_{stance_file}.npz"
    text_file_path = os.path.join(arg.data_dir, stance_input_dir)
    TEXT_emb_dict = get_text_embeddings_dict(text_file_path)
    price_data_path = os.path.join(arg.data_dir, "price_data_ectsum")
    vol_single_df = pd.read_csv(
        os.path.join(price_data_path, "ectsum_SeriesSingleDayVol3.csv")
    )
    vol_average_df = pd.read_csv(
        os.path.join(price_data_path, "ectsum_Avg_Series_WITH_LOG.csv")
    )
    price_df = pd.read_csv(os.path.join(price_data_path, "ectsum_price_label.csv"))
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
            f"future_label{arg.duration}",
            "current_adjclose_price",
        ]
    ]
    vol_single_df = vol_single_df[["text_file_name", f"future_Single_{arg.duration}"]]

    vol_average_df = vol_average_df[["text_file_name", f"future_Single_{arg.duration}"]]
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
        LABEL_emb = merged_data[f"future_label{arg.duration}"].values
    elif arg.task == "volatility_prediction":
        merged_data = pd.merge(
            vol_single_df, vol_average_df, on="text_file_name", how="inner"
        )

        LABEL_emb = merged_data[f"future_Single_{arg.duration}_x"].values
    elif arg.task == "stock_return_prediction":
        merged_data = pd.merge(
            vol_single_df, price_df, on="text_file_name", how="inner"
        )
        LABEL_emb = merged_data[f"stock_return_{arg.duration}"].values
    else:
        raise ValueError("task is not well defined")
    merged_text_file_list = merged_data["text_file_name"].tolist()
    TEXT_emb = stack_text_embeddings(TEXT_emb_dict, merged_text_file_list)
    various_dataset = Dataset_single_task(TEXT_emb, LABEL_emb, merged_text_file_list)
    various_dataloader = torch.utils.data.DataLoader(
        various_dataset,
        batch_size=len(various_dataset),
        shuffle=False,
        num_workers=2,
    )

    # if majority_vote > 0.5:
    #     result_array = np.ones_like(LABEL_emb)
    # else:
    #     result_array = np.zeros_like(LABEL_emb)
    # print(f"accuracy of majority vote: {accuracy_score(LABEL_emb, result_array)}")
    # print(f"precision of majority vote: {precision_score(LABEL_emb, result_array)}")
    # print(f"recall of majority vote: {recall_score(LABEL_emb, result_array)}")

    return various_dataloader


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


def calculate_log_volatility(row, start, end):
    # Extract the current adjusted close price
    current_price = row["current_adjclose_price"]

    # Collect future prices into a list
    future_prices = [current_price] + [
        row[f"future_{i}"] for i in range(start, end + 1) if f"future_{i}" in row
    ]

    # Calculate logarithmic returns from the current price to each future price
    returns = np.array(
        [
            (future_prices[i + 1] / future_prices[i]) - 1
            for i in range(len(future_prices) - 1)
        ]
    )

    # Calculate the mean of logarithmic returns
    mean_return = np.mean(returns)

    # Calculate the standard deviation of returns normalized by their mean
    delta = len(returns)
    volatility = np.sqrt(np.sum((returns - mean_return) ** 2) / delta)

    # Return the natural logarithm of the calculated volatility
    log_volatility = np.log(volatility)
    return log_volatility


def go(arg):
    """
    Creates and trains a basic transformer for the volatility regression task.
    """
    LOG2E = math.log2(math.e)
    NUM_CLS = 1

    print(" Loading Data ...")
    # Text Embeddings Load
    # input_dirはECTかGpt_summary
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
        # val: np.ndarray, (56, 520, 768)
        # val_label: np.ndarray, (56,)
        # text_file_val_label: list, (56,)
        val, val_label, text_file_val_label = val_data

        # majorityの実装
        majority_vote = np.mean(train_label)
        print(f"majority vote: {majority_vote}")
        # if majority_vote > 0.5:
        #     result_array = np.ones_like(val_label)
        # else:
        #     result_array = np.zeros_like(val_label)
        # print(f"accuracy of majority vote: {accuracy_score(val_label, result_array)}")
        # print(f"precision of majority vote: {precision_score(val_label, result_array)}")
        # print(f"recall of majority vote: {recall_score(val_label, result_array)}")
        # import ipdb

        # ipdb.set_trace()
        training_set = Dataset_single_task(train, train_label, text_file_train_label)
        val_set = Dataset_single_task(val, val_label, text_file_val_label)

    if arg.train_normal_test_various:
        assert arg.file_name in [
            "TextSequence",
            "ECT",
            "gpt4_summary",
        ], "TextSequence only"
        assert arg.final == False, "Not Final"
        various_test_set = {}
        for stance_file in GPT_FILE_TYPES:
            if arg.test_type == "acl19":
                various_dataloader = get_acl_dataloader(
                    arg,
                    stance_file,
                    merged_text_file_list,
                    LABEL_emb,
                    text_file_val_label,
                )
            elif arg.test_type == "ectsum":
                # ECTSUM用のvarious dataloader，textembとlabelemb両方新しいのにする
                various_dataloader = get_ectsum_dataloader(arg, stance_file)
            else:
                raise ValueError("test_type is not well defined")
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
