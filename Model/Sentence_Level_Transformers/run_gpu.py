import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import ipdb

# ipdb.set_trace()
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


class Dataset(VanillaDataset):
    def __init__(self, texts, labels, labels_b):
        "Initialization"
        self.labels = labels
        self.text = texts
        self.labels_b = labels_b

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
        y_b = self.labels_b[index]
        return X, y, y_b


def go(arg):
    """
    Creates and trains a basic transformer for the volatility regression task.
    """
    LOG2E = math.log2(math.e)
    NUM_CLS = 1

    print(" Loading Data ...")
    TEXT_emb = np.load(arg.data_dir + arg.input_dir, allow_pickle=True)
    TEXT_emb_dict = {key: TEXT_emb[key] for key in TEXT_emb}
    # TODO stock priceとvolatilityのデータロード
    price_data_path = arg.data_dir + arg.price_data_dir

    vol_single_path = [
        "train_split_SeriesSingleDayVol3.csv",
        "val_split_SeriesSingleDayVol3.csv",
        "test_split_SeriesSingleDayVol3.csv",
    ]
    vol_average_path = [
        "train_split_Avg_Series_WITH_LOG.csv",
        "val_split_Avg_Series_WITH_LOG.csv",
        "test_split_Avg_Series_WITH_LOG.csv",
    ]
    price_path = [
        "train_price_label.csv",
        "dev_price_label.csv",
        "test_price_label.csv",
    ]

    vol_single_list = []
    vol_average_list = []
    price_list = []
    for vol_single, vol_average, price in zip(
        vol_single_path, vol_average_path, price_path
    ):
        vol_single_list.append(pd.read_csv(price_data_path + vol_single))
        vol_average_list.append(pd.read_csv(price_data_path + vol_average))
        price_list.append(pd.read_csv(price_data_path + price))

    vol_single_df = pd.concat(vol_single_list, axis=0)
    vol_average_df = pd.concat(vol_average_list, axis=0)
    price_df = pd.concat(price_list, axis=0)

    # Multi-task on volatility average and volatility single <- 3 days
    text_file_list = list(TEXT_emb_dict.keys())
    vol_single_df = vol_single_df[vol_single_df["text_file_name"].isin(text_file_list)]
    vol_average_df = vol_average_df[
        vol_average_df["text_file_name"].isin(text_file_list)
    ]
    price_df = price_df[price_df["text_file_name"].isin(text_file_list)]

    vol_single_df = vol_single_df[
        ["text_file_name", f"future_Single_{arg.vol_duration}"]
    ]
    vol_average_df = vol_average_df[["text_file_name", f"future_{arg.vol_duration}"]]

    # TODO stock priceとtext_embのkeyが一致しているように確認
    merged_data = pd.merge(
        vol_single_df, vol_average_df, on="text_file_name", how="inner"
    )
    LABEL_emb = merged_data[f"future_{arg.vol_duration}"].values
    LABEL_emb_b = merged_data[f"future_Single_{arg.vol_duration}"].values
    TEXT_emb = np.stack(
        [TEXT_emb_dict[i] for i in merged_data["text_file_name"].tolist()]
    )
    print(" Finish Loading Data... ")

    if arg.final:

        train, test = train_test_split(TEXT_emb, test_size=0.2)
        train_label, test_label = train_test_split(LABEL_emb, test_size=0.2)
        train_label_b, test_label_b = train_test_split(LABEL_emb_b, test_size=0.2)

        training_set = Dataset(train, train_label, train_label_b)
        val_set = Dataset(test, test_label, test_label_b)

    else:
        data, _ = train_test_split(TEXT_emb, test_size=0.2)
        train, val = train_test_split(data, test_size=0.125)

        data_label, _ = train_test_split(LABEL_emb, test_size=0.2)
        train_label, val_label = train_test_split(data_label, test_size=0.125)

        data_label_b, _ = train_test_split(LABEL_emb_b, test_size=0.2)
        train_label_b, val_label_b = train_test_split(data_label_b, test_size=0.125)

        training_set = Dataset(train, train_label, train_label_b)
        val_set = Dataset(val, val_label, val_label_b)

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
    model = RTransformer(
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
        "Test Loss B": [],
        "Outputs": [],
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

            inputs, labels, labels_b = data
            inputs = Variable(inputs.type(torch.FloatTensor))
            labels = torch.tensor(labels, dtype=torch.float32).cuda()
            labels_b = torch.tensor(labels_b, dtype=torch.float32).cuda()
            # if i ==0:
            # print (inputs.shape)
            if inputs.size(1) > arg.max_length:
                inputs = inputs[:, : arg.max_length, :]

            out_a, out_b = model(inputs)
            # print(out_a.shape,out_b.shape)
            # print(out.shape,labels.shape)

            loss_a = F.mse_loss(out_a, labels)
            loss_b = F.mse_loss(out_b, labels_b)
            loss = arg.alpha * loss_a + (1 - arg.alpha) * loss_b
            train_loss_tol += loss

            loss.backward()

            # clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if arg.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

            opt.step()

            seen += inputs.size(0)
            # tbw.add_scalar('classification/train-loss', float(loss.item()), seen)
        # print('train_loss: ',train_loss_tol)
        train_loss_tol = train_loss_tol / (i + 1)
        with torch.no_grad():

            model.train(False)
            tot, cor = 0.0, 0.0

            loss_test = 0.0
            loss_test_b = 0.0
            for i, data in enumerate(testloader):
                inputs, labels, labels_b = data
                inputs, labels, labels_b = (
                    torch.tensor(inputs, dtype=torch.float32),
                    torch.tensor(labels, dtype=torch.float32).cuda(),
                    torch.tensor(labels_b, dtype=torch.float32).cuda(),
                )
                if inputs.size(1) > arg.max_length:
                    inputs = inputs[:, : arg.max_length, :]
                out_a, out_b = model(inputs)

                loss_test += F.mse_loss(out_a, labels)
                loss_test_b += F.mse_loss(out_b, labels_b)
                # tot = float(inputs.size(0))
                # cor += float(labels.sum().item())

            acc = loss_test
        #             if arg.final:
        #                 print('test accuracy', acc)
        #             else:
        #                 print('validation accuracy', acc)
        # torch.save(model, '/data/exp/checkpoints_torch_volatility/checkpoint-epoch'+str(e)+'.pth')
        evaluation["epoch"].append(e)
        evaluation["Train Loss"].append(train_loss_tol.item())
        evaluation["Test Loss"].append(acc.item())
        evaluation["Test Loss B"].append(loss_test_b.item())
        evaluation["Outputs"].append(out_a.cpu().detach().numpy().mean())

    evaluation = pd.DataFrame(evaluation)
    evaluation.sort_values(["Test Loss"], ascending=True, inplace=True)

    return evaluation
