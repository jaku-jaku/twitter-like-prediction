"""
This main would predict likes based on the tweets only
"""
# %% Import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from ast import literal_eval
import os
import sys
import json

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import re
import emoji
import operator
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import datetime

# ML:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# debugger:
from icecream import ic

import jieba # to split east-asian language to words

## USER DEFINED:
ABS_PATH = "/home/jx/JXProject/Github/UW__4B_Individual_Works/CS 480/Kaggle" # Define ur absolute path here

## Custom Files:
def abspath(relative_path):
    return os.path.join(ABS_PATH, relative_path)

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(abspath("src_code"))

# Custom Lib
import jx_lib
from jx_pytorch_lib import ProgressReport

# %% LOAD DATASET: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# import data
TRAIN_DATA_X = pd.read_csv(abspath("data/p_train_x.csv"))
TRAIN_DATA_Y = pd.read_csv(abspath("data/p_train_y.csv"))
TEST_DATA_X = pd.read_csv(abspath("data/p_test_x.csv"))
ic(np.sum(TRAIN_DATA_X["id"] == TRAIN_DATA_Y["id"])) # so the assumption should be right, they share the exact same id in sequence
TRAIN_DATA = pd.concat([TRAIN_DATA_X, TRAIN_DATA_Y["likes_count"]], axis=1)
ic(TRAIN_DATA.shape)

# %% Pre-Data Analysis: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# ANALYSIS_OUTPUT_FOLDER =  abspath("output-analysis")
# jx_lib.create_folder(ANALYSIS_OUTPUT_FOLDER)

# TRAIN_DATA.head(10)
# sns.displot(TRAIN_DATA["likes_count"])
# HEADERS = list(TRAIN_DATA.columns)
# a = ic(HEADERS)

# # Plot Language and video Count:
# fig = plt.figure(figsize=(20,20))
# ax = plt.subplot(2, 2, 1)
# ax.set_title("Language Count")
# sns.histplot(ax=ax, data=TRAIN_DATA, x="language", hue="likes_count", multiple="dodge")
# fig.savefig("{}/plot_{}.png".format(ANALYSIS_OUTPUT_FOLDER, "preprocess"), bbox_inches = 'tight')

# ax = plt.subplot(2, 2, 2)
# ax.set_title("Video Count")
# sns.histplot(ax=ax, data=TRAIN_DATA, x="video", hue="likes_count", multiple="dodge")
# fig.savefig("{}/plot_{}.png".format(ANALYSIS_OUTPUT_FOLDER, "preprocess"), bbox_inches = 'tight')

# %% -------------------------------- CUSTOM NETWORK LIBRARY -------------------------------- %% #
# # DEFINE NETWORK: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
"""
Logistic Regression Bag-of-Words classifier:

- Have tried custom deeper network => takes too long to train and not very effective
- The best is still one layer FN, aka logistic regression
- The best we may achiever: {see crowdmark}
- we will split training set into 90\% training and 10\% validation
"""
class BOW_Module(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BOW_Module, self).__init__()
        # the parameters of the affine mapping.
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        # input => Linear => softmax
        return F.log_softmax(self.linear(bow_vec), dim=1)

class BOW_ModuleV2(nn.Module):
    def __init__(self, num_labels, vocab_size, 
        dropout=0.2, d_hidden=100, n_layers=2,
    ):
        super(BOW_ModuleV2, self).__init__()
        self.linear1 = nn.Linear(vocab_size, d_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_hidden, num_labels)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, bow_vec):
        # input => Linear => softmax
        y = self.linear1(bow_vec)
        # y = self.dropout(y)
        y =self.relu(y)
        y = self.linear2(y)
        outputs =  F.log_softmax(y, dim=1)
        return outputs

class BOW_ModuleV2Drop(nn.Module):
    def __init__(self, num_labels, vocab_size, 
        dropout=0.2, d_hidden=100, n_layers=2,
    ):
        super(BOW_ModuleV2Drop, self).__init__()
        K_size = 10
        N_stride = 5
        self.linear1 = nn.Linear(vocab_size, d_hidden)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, num_labels)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, bow_vec):
        # input => Linear => softmax
        y = self.linear1(bow_vec)
        y = self.relu(y)
        y = self.drop(y)
        y = self.linear2(y)
        outputs =  F.log_softmax(y, dim=1)
        return outputs

class BOW_ModuleV3(nn.Module):
    def __init__(self, num_labels, vocab_size, 
        dropout=0.2, d_hidden=1000, n_layers=2,
    ):
        super(BOW_ModuleV3, self).__init__()
        self.linear1 = nn.Linear(vocab_size, d_hidden)
        self.linear2 = nn.Linear(d_hidden, 100)
        self.linear3 = nn.Linear(100, num_labels)
        self.relu = nn.ReLU()

    def forward(self, bow_vec):
        # input => Linear => softmax
        y = self.linear1(bow_vec)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.relu(y)
        y = self.linear3(y)
        outputs =  F.log_softmax(y, dim=1)
        return outputs

class TwitterLikePredictor:
    # %% generate a word to index:
    word_to_ix = {}
    word_count = {}
    label_to_ix = {}
    NUM_LABELS = 0
    VOCAB_SIZE = 0

    @dataclass
    class PredictorConfiguration:
        Y_TAG                : str              = "likes_count"
        USE_GPU              : bool             = True
        MODEL_TAG            : str              = "default"
        # tweet pre-processing: 
        PRE_PROCESS_TAG      : str              = "default-with-username"
        DELIMITER_SET        : str              = "; |, |、|。| "
        SYMBOLE_REMOVE_LIST  : str              = field(default_factory=lambda: ["\[", "\]", "\(", "\)"])
        KEYS_TO_REMOVE_LIST  : List             = field(default_factory=lambda: ["http", "arXiv", "https"])
        ENABLE_EXTRA_CONVERSION  : List         = field(default_factory=lambda: [])
        # training set:
        SHUFFLE_TRAINING     : bool             = False
        PERCENT_TRAINING_SET : float            = 0.9
        # bag of words (Tweeter Interpretation):
        BOW_TOTAL_NUM_EPOCHS : int              = 10
        LOSS_FUNC            : nn               = nn.NLLLoss()
        LEARNING_RATE        : float            = 0.001
        FORCE_REBUILD        : bool             = False 
        OPTIMIZER            : float            = optim.SGD
        MOMENTUM             : float            = 0.9
        MODEL_VERSION        : str              = "v2"
        D_HIDDEN             : int              = 100
        N_TOP_FEATURES       : Optional[int]    = None
        N_MIN_VARIANCE       : Optional[int]    = None
        N_EARLY_STOPPING_NDROPS : Optional[int] = 3

    def _print(self, content):
        if self.verbose:
            print("[TLP] ", content)
            # output to file
        with open(os.path.join(self.folder_dict["output"],"TLP_log.txt"), "a") as log_file:
            log_file.write("\n")
            log_file.write(content)

    def __init__(
        self, 
        pd_data_training,
        pd_data_testing,
        engine_name,
        verbose,
        verbose_show_sample_language_parse,
        config: PredictorConfiguration
    ):
        t_init = time.time()
        self.config = config
        self.verbose = verbose
        # autogen folder:
        self.folder_dict = {
            "processed_data": "processed_data", # store processed dataset data
            "output-dir"    : "output",
            "output"        : "output/{}".format(engine_name),
            "y-pred"        : None,
            "models"        : None,
            "analysis"      : None
        }
        for name_ , folder_name_ in self.folder_dict.items():
            if folder_name_ is not None:
                abs_path = abspath(folder_name_)
            else:
                abs_path = abspath("{}/{}/{}".format("output", engine_name, name_))
            jx_lib.create_folder(abs_path)
            self.folder_dict[name_] = abs_path
        # print config:
        self._print(str(self.config))
        # Prepare Hardware ==== ==== ==== ==== ==== ==== ==== #
        self._print("Prepare Hardware")
        if self.config.USE_GPU:
            self.device = self.load_device()
        # Pre-processing Dataset ==== ==== ==== ==== ==== ==== ==== #
        self._print("Pre-processing Dataset ...")
        path_test_lite = os.path.join(self.folder_dict["processed_data"], 'preprocessed-lite-test-[{}].csv'.format(self.config.PRE_PROCESS_TAG))
        path_test_preprocessed = os.path.join(self.folder_dict["processed_data"], 'preprocessed-idx-test-[{}].csv'.format(self.config.PRE_PROCESS_TAG))
        path_lite = os.path.join(self.folder_dict["processed_data"], 'preprocessed-lite-train-[{}].csv'.format(self.config.PRE_PROCESS_TAG))
        path_preprocessed = os.path.join(self.folder_dict["processed_data"], 'preprocessed-idx-train-[{}].csv'.format(self.config.PRE_PROCESS_TAG))
        path_dict = os.path.join(self.folder_dict["processed_data"], 'bow-dict-[{}].json'.format(self.config.PRE_PROCESS_TAG))
        self.if_read_from_file = False
        if os.path.exists(path_preprocessed) and not self.config.FORCE_REBUILD:
            # load training:
            self.training_dataset = pd.read_csv(path_preprocessed)
            self._print("Loading Pre-Processed Dataset From {} [size:{}]".format(path_preprocessed, self.training_dataset.shape))
            self.pytorch_data_train_id, self.pytorch_data_eval_id,\
                self.pytorch_data_train_preprocessed, self.pytorch_data_eval_preprocessed = self.split_training_dataset(
                x_tag = "norm-tweet-bow-idx-array",
                pd_data = self.training_dataset,
                config = self.config,
                if_literal_eval = True
            )
            # load test:
            self._print("Loading Pre-Processed Test Dataset From {}".format(path_test_preprocessed))
            self.final_testing_dataset = pd.read_csv(path_test_preprocessed)
            self.if_read_from_file = True
            # load json wov dictionary
            self._print("Loading Pre-Processed Word To Vector IX From {}".format(path_dict))
            with open(path_dict, "r") as f:
                self.word_to_ix = json.load(f)
            # gen const:
            labels = self.training_dataset[self.config.Y_TAG].unique()
            self.VOCAB_SIZE = len(self.word_to_ix)
            self.NUM_LABELS = len(labels)
        else:
            # Pre-processing Sentences ==== ==== ==== ==== ==== ==== ==== #
            # break sentebce to bag of words:
            # training data:
            self._print("No previous cache found, Let's Pre-process Dataset ...")
            self.training_dataset = self.generate_tweet_message_normalized_column(pd_data=pd_data_training, config=self.config)
            self.training_dataset.to_csv(path_lite)
            self._print("Pre-processed Training Dataset (Lite) Saved @ {}".format(path_lite))
            # testing data:
            self.final_testing_dataset = self.generate_tweet_message_normalized_column(pd_data=pd_data_testing, config=self.config)
            self.final_testing_dataset.to_csv(path_test_lite)
            self._print("Pre-Processed Test Dataset (Lite) Saved @ {}".format(path_test_lite))
            # sample:
            UNIQ_LANG =  self.training_dataset["language"].unique().tolist()
            if verbose_show_sample_language_parse:
                for lang in UNIQ_LANG:
                    index =  self.training_dataset.index[ self.training_dataset["language"] == lang].tolist()[0]
                    self._print("{} > {}".format(lang,  self.training_dataset["norm-tweet"][index]))
            labels = self.training_dataset[self.config.Y_TAG].unique()
            self.label_to_ix = {i:i for i in labels}
            self.NUM_LABELS = len(labels)
            # Prepare Dataset ==== ==== ==== ==== ==== ==== ==== #
            self._print("Prepare Training Dataset")
            self.pytorch_data_train_id, self.pytorch_data_eval_id, \
                pytorch_data_train, pytorch_data_eval = self.split_training_dataset(
                    x_tag = "norm-tweet",
                    pd_data = self.training_dataset,
                    config = self.config
                )
            self.final_testing_dataset["likes_count"] = [0] * len(self.final_testing_dataset) # MOCK / UNUSED
            _, _, data1, data2 = self.split_training_dataset(
                    x_tag = "norm-tweet",
                    pd_data = self.final_testing_dataset,
                    config = self.config
                )
            # Generate Bag of Words ==== ==== ==== ==== ==== ==== ==== #
            self._generate_bow_dictionary(
                data = pytorch_data_train + pytorch_data_eval, #+ data1 + data2, # based on all possible data
                n_top_features = self.config.N_TOP_FEATURES,
                n_min_variance = self.config.N_MIN_VARIANCE,
            )
            f = open(path_dict, "w")
            json.dump(self.word_to_ix, f)
            f.close()

            # Pre-process Dataset (Word => vector) ==== ==== ==== ==== ==== #
            self._print("Prepare Training Dataset Pre-Vectorization")
            # training:
            self.training_dataset = self.generate_tweet_message_normalized_column_converted(pd_data=self.training_dataset)
            self.training_dataset.to_csv(path_preprocessed)
            self._print("Pre-processed Dataset Again (idx) Saved @ {}".format(path_preprocessed))
            # final testing:
            self.final_testing_dataset = self.generate_tweet_message_normalized_column_converted(pd_data = self.final_testing_dataset)
            self.final_testing_dataset.to_csv(path_test_preprocessed)
            self._print("Pre-Processed Final Test Dataset Again (idx) Saved @ {}".format(path_test_preprocessed))
            # split training dataset again for precompiled dataset:
            self.pytorch_data_train_id, self.pytorch_data_eval_id,\
                self.pytorch_data_train_preprocessed, self.pytorch_data_eval_preprocessed = self.split_training_dataset(
                x_tag = "norm-tweet-bow-idx-array",
                pd_data = self.training_dataset,
                config = self.config,
                train_id = self.pytorch_data_train_id, 
                eval_id = self.pytorch_data_eval_id
            )
        # give a brief summary of number of training sets
        self._print("N_train:{}  N_validation:{}".format(
            len(self.pytorch_data_train_preprocessed), 
            len(self.pytorch_data_eval_preprocessed)
        ))
        # Init Model ==== ==== ==== ==== ==== ==== ==== ==== #
        self._print("New Model Created")
        self.create_new_model(version=self.config.MODEL_VERSION)
        if self.config.USE_GPU:
            self.model.to(self.device)
        # REPORT ==== ==== ==== ==== ==== ==== ==== ==== === #
        self._print("REPORT SUMMARY ======================= ")
        self._print("VOCAB_SIZE: {}".format(self.VOCAB_SIZE))
        self._print("NUM_LABELS: {}".format(self.NUM_LABELS))
        # print model parameters
        self._print("MODEL: {}".format(self.model))
        t_init = time.time() - t_init
        self._print("======================= END OF INIT [Ellapsed:{}s] =======================".format(t_init))
    
    @staticmethod
    def generate_tweet_message_normalized_column(
        pd_data, 
        config
    ):
        MAX_LENGTH, d = pd_data.shape
        tweet_data = []
        for i in range(MAX_LENGTH):
            # include name and tweet for token analysis
            messages = str(pd_data['name'][i])
            len_tweet = len(pd_data['tweet'][i])
            if "no-tweet" not in config.ENABLE_EXTRA_CONVERSION:
                messages = " " + pd_data['tweet'][i]
            # remove some characters with space
            for sym in config.SYMBOLE_REMOVE_LIST:
                messages = re.sub(sym, " ", messages)
            # separate delimiter:
            messages = re.split(config.DELIMITER_SET, messages)
            # split emojis
            new_messages = []
            for msg in messages:
                new_messages.extend(emoji.get_emoji_regexp().split(msg))
            messages = new_messages
            # remove keys:
            new_messages = []
            for msg in messages:
                no_key = True
                for key in config.KEYS_TO_REMOVE_LIST: # tags to be removed
                    if key in msg:
                        msg = key # no_key = False, lets replace key with key name => as feature
                if no_key and len(msg) > 0:
                    # split:
                    new_messages.extend(jieba.lcut(msg, cut_all=True)) # split east asian
            # convert emojis:
            if "unify-emoji" in config.ENABLE_EXTRA_CONVERSION:
                new_messages = ["[emoji]" if emoji.get_emoji_regexp().search(msg) else msg for msg in new_messages]
            # by anaylsis, we found significant correlation in time, user_id ..., and try emphasis by repeating the tokens
            # Let's convert time and other correlation into word as well!
            time_str = pd_data["created_at"][i]
            time_str = time_str.split(" ")
            date_ = datetime.datetime.strptime(time_str[0], "%Y-%m-%d")
            time_ = datetime.datetime.strptime(time_str[1], "%H:%M:%S")
            descriptive_str = []
            time_str_set = [
                # time related:
                "[year:{}]".format(date_.year),
                "[month:{}]".format(date_.month),
                "[day:{}]".format(date_.day),
                "[hour:{}]".format(time_.hour),
                "[zone:{}]".format(time_str[2]),
            ]
            descriptive_str.extend(time_str_set * 5) 
            # existance of other placeholders
            placeholders_str = []
            if (not pd.isna(pd_data["place"][i])):
                placeholders_str.append("[exist:place]")
            if (not pd.isna(pd_data["quote_url"][i])):
                placeholders_str.append("[exist:quote_url]")
            if (not pd.isna(pd_data["thumbnail"][i])):
                placeholders_str.append("[exist:thumbnail]")
            
            # not effective based on analysis
            # n_replyto = len(literal_eval(pd_data["reply_to"][i])) > 0
            # placeholders_str.extend(["[exist:reply:to]"] * n_replyto)     
            descriptive_str.extend(placeholders_str)
            
            # depends on tweet length 0~30       
            placeholders_str.append("[length:tweet:{}]".format(int(len_tweet))) 
            # include langage
            descriptive_str.append("[lang:{}]".format(pd_data['language'][i]))
            # include hashtags: (should already be inside the tweet, but let's emphasize it by repeating)
            descriptive_str.extend(literal_eval(pd_data['hashtags'][i]))
            # include user_id (emphasize x10)
            descriptive_str.extend(["[user:{}]".format(pd_data['user_id'][i])] * 20)
            
            # extend messages
            new_messages.extend(descriptive_str)
            # append to normalized tweet data
            tweet_data.append(new_messages)

        pd_data['norm-tweet'] = tweet_data
        return pd_data


    def create_new_model(self, version="v2"):
        self._print("Use Model Version: {}".format(version))
        self.version = version
        if version == "v2":
            # self.model = nn.Sequential(
            #     nn.Linear(self.VOCAB_SIZE, self.NUM_LABELS),
            #     nn.Softmax(dim=1)
            # )
            self.model = BOW_ModuleV2(self.NUM_LABELS, self.VOCAB_SIZE, d_hidden=self.config.D_HIDDEN)
        elif version == "v3":
            self.model = BOW_ModuleV3(self.NUM_LABELS, self.VOCAB_SIZE, d_hidden=self.config.D_HIDDEN)
        elif version == "v2-drop":
            self.model = BOW_ModuleV2Drop(self.NUM_LABELS, self.VOCAB_SIZE, d_hidden=self.config.D_HIDDEN)
        else:
            self.model = BOW_Module(self.NUM_LABELS, self.VOCAB_SIZE)

    def generate_tweet_message_normalized_column_converted(self, pd_data):
        column = []
        for bow_ in pd_data['norm-tweet']:
            column.append(self.make_bow_idx_array(bow_))
        pd_data['norm-tweet-bow-idx-array'] = column
        return pd_data

    @staticmethod
    def load_device():
        # hardware-acceleration
        device = None
        if torch.cuda.is_available():
            print("[ALERT] Attempt to use GPU => CUDA:0")
            device = torch.device("cuda:0")
        else:
            print("[ALERT] GPU not found, use CPU!")
            device =  torch.device("cpu")
        return device
    
    # gen pytorch data:
    @staticmethod
    def pandas2pytorch(
        pd_data,
        x_tag: str,
        y_tag: str,
        range: List[int],
        if_literal_eval: bool = False
    ) -> "id, training pair":
        id_,x_,y_ = pd_data['id'][range[0]:range[1]], pd_data[x_tag][range[0]:range[1]], pd_data[y_tag][range[0]:range[1]]
        if if_literal_eval:
            return id_.tolist(), [(literal_eval(x),y) for x,y in zip(x_,y_)]
        else:
            return id_.tolist(), [(x,y) for x,y in zip(x_,y_)]


    @staticmethod
    def split_training_dataset(
        pd_data, config, x_tag,
        train_id = None, eval_id = None,
        if_literal_eval: bool = False
    ):
        if train_id is not None and eval_id is not None:
            pytorch_data_train = [(pd_data[x_tag][id_], pd_data[config.Y_TAG][id_]) for id_ in train_id]
            pytorch_data_eval = [(pd_data[x_tag][id_], pd_data[config.Y_TAG][id_]) for id_ in eval_id]
        else:
            # let's shuffle the training data:
            if config.SHUFFLE_TRAINING:
                pd_data = pd_data.sample(frac = 1)
            
            if config.PERCENT_TRAINING_SET is None:
                N_TRAIN = len(pd_data)
                N_TEST = 0
            else:
                N_TRAIN = int(len(pd_data) * config.PERCENT_TRAINING_SET)
                N_TEST = len(pd_data) - N_TRAIN
            # let's split data
            train_id,pytorch_data_train = TwitterLikePredictor.pandas2pytorch(
                pd_data = pd_data,
                x_tag = x_tag, y_tag = config.Y_TAG,
                range =[0, N_TRAIN],
                if_literal_eval = if_literal_eval
            )
            eval_id,pytorch_data_eval = TwitterLikePredictor.pandas2pytorch(
                pd_data = pd_data,
                x_tag = x_tag, y_tag = config.Y_TAG,
                range =[N_TRAIN, N_TRAIN+N_TEST],
                if_literal_eval = if_literal_eval
            )
        return train_id, eval_id, pytorch_data_train, pytorch_data_eval

    def _generate_bow_dictionary(
        self,
        data,
        n_top_features = None,
        n_min_variance = None,
    ):
        for sent, y in data:
            for word in sent:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
                    self.word_count[word] = [0] * self.NUM_LABELS
                else:
                    self.word_count[word][y] += 1
        self.VOCAB_SIZE = len(self.word_to_ix)
        # sort tokens based on the variance:
        self.word_count_top_features = sorted(self.word_count.items(), key=lambda x:-np.var(x[1]))
        # Feature reduction
        if n_min_variance is not None:
            self._print("Filter Requsted: var:{}".format(n_min_variance))
            # filter by variance:
            self.word_count_top_features = [x for x in self.word_count_top_features if np.var(x[1]) >= n_min_variance]
            # sample top bag of words
            self.word_to_ix = {}
            for word, count in self.word_count_top_features:
                self.word_to_ix[word] = len(self.word_to_ix)
            self.VOCAB_SIZE = len(self.word_to_ix)

        if n_top_features is not None:
            self._print("Reduction Requsted: {}->{}".format(self.VOCAB_SIZE, n_top_features))
            # sample top bag of words
            self.word_count_top_features = self.word_count_top_features[:n_top_features]
            self.word_to_ix = {}
            for word in self.word_count_top_features:
                self.word_to_ix[word[0]] = len(self.word_to_ix)
            self.VOCAB_SIZE = len(self.word_to_ix)
    
    def make_bow_idx_array(self, sentence):
        vec = []
        for word in sentence:
            # do not use word if it was not in the dictionary, this happens when unseen testing dataset
            if word in self.word_to_ix:
                vec.append(self.word_to_ix[word])
        return vec

    def convert_bow_idx_array_2_vector(self, bow_idx_array):
        vec = torch.zeros(self.VOCAB_SIZE)
        for idx in bow_idx_array:
            vec[idx] += 1
        return vec.view(1, -1)

    def make_target(self, label):
        return torch.LongTensor([label])

    def train(self, gen_plot:bool=True, sample_threshold:float=0.5):
        self._print("\n\nTRAING BEGIN -----------------------------:")
        report_ = ProgressReport()
        if self.config.OPTIMIZER == optim.SGD:
            optimizer_ = self.config.OPTIMIZER(
                self.model.parameters(), lr=self.config.LEARNING_RATE, momentum=self.config.MOMENTUM)
        else:
            optimizer_ = self.config.OPTIMIZER(self.model.parameters(), lr=self.config.LEARNING_RATE)

        loss_ = self.config.LOSS_FUNC

        n_accuracy_drops = 0
        for epoch in range(self.config.BOW_TOTAL_NUM_EPOCHS):
            self._print("> epoch {}/{}:".format(epoch + 1, self.config.BOW_TOTAL_NUM_EPOCHS))
    
            train_loss_sum, train_acc_sum, train_n, train_start = 0.0, 0.0, 0, time.time()
            val_loss_sum, val_acc_sum, val_n, val_start = 0.0, 0.0, 0, time.time()

            # TRAIN -----------------------------:
            i, n = 0, len(self.pytorch_data_train_preprocessed)
            for instance, label in self.pytorch_data_train_preprocessed:
                i += 1
                print("\r > Training [{}/{}]".format(i, n),  end='')
                # 1: Clear PyTorch Cache
                self.model.zero_grad()
                # 2: Convert BOW to vectors:
                bow_vec = self.convert_bow_idx_array_2_vector(instance)
                target =  self.make_target(label)
                # 3: fwd:
                if self.config.USE_GPU:
                    bow_vec = bow_vec.to(self.device)
                    target = target.to(self.device)
                log_probs = self.model(bow_vec)

                # 4: backpropagation (training)
                loss = loss_(log_probs, target)
                loss.backward()
                optimizer_.step()
                # Log summay:
                train_loss_sum += loss.item()
                train_acc_sum += (log_probs.argmax(dim=1) == label).sum().item()
                # train_acc_sum += (log_probs.argmax(dim=1) == yi).sum().item()
                train_n += 1
            
            train_ellapse = time.time() - train_start
            print("\n",  end='')
    
            # TEST -----------------------------:
            i, n = 0, len(self.pytorch_data_eval_preprocessed)
            if n > 0:
                with torch.no_grad(): # Not training!
                    for instance, label in self.pytorch_data_eval_preprocessed:
                        i += 1
                        print("\r > Validating [{}/{}]".format(i, n),  end='')
                        # bow_vec = self.make_bow_vector(instance)
                        # target = self.make_target(label)
                        bow_vec = self.convert_bow_idx_array_2_vector(instance)
                        target =  self.make_target(label)
                        if self.config.USE_GPU:
                            bow_vec = bow_vec.to(self.device)
                            target = target.to(self.device)
                        log_probs = self.model(bow_vec)
                        # Log summay:
                        loss = loss_(log_probs, target)
                        val_loss_sum += loss.item()
                        val_acc_sum += (log_probs.argmax(dim=1) == label).sum().item()
                        val_n += 1
            else:
                val_n = 1
                val_acc_sum = 0

            val_ellapse = time.time() - val_start
            print("\n",  end='')

            val_acc = val_acc_sum / val_n
            if epoch > 1 and report_.history["test_acc"][-1] > val_acc:
                n_accuracy_drops += 1
            else:
                n_accuracy_drops = 0

            # Log ------:
            self._print(report_.append(
                epoch         = epoch,
                train_loss    = train_loss_sum / train_n,
                train_acc     = train_acc_sum / train_n,
                train_time    = train_ellapse,
                test_loss     = val_loss_sum / val_n,
                test_acc      = val_acc,
                test_time     = val_ellapse,
                learning_rate = 0,
            ))

            if (val_acc >= sample_threshold): 
                # early sampling, save model that meets minimum threshold
                self._print("> [Minimum Goal Reached] Attempt to predict, with {}>={}:".format(val_acc, sample_threshold))
                tag = "autosave-e:{}".format(epoch)
                df_pred = TLP_Engine.predict(tag=tag)
                self.save_model(tag=tag)
                sample_threshold = val_acc # keep recording for better ones
            
            elif (self.config.N_EARLY_STOPPING_NDROPS is None and np.mod(epoch, 10) == 0): 
                # fixed sampling per 10 epoch
                self._print("> [Per 10 epoch Auto-sampling] {}:".format(val_acc))
                tag = "autosave-per10-e:{}".format(epoch)
                df_pred = TLP_Engine.predict(tag=tag)
                # self.save_model(tag=tag)
            
            if self.config.N_EARLY_STOPPING_NDROPS is not None:
                if (n_accuracy_drops >= self.config.N_EARLY_STOPPING_NDROPS):
                    self._print("> Early Stopping due to accuracy drops in last {} iterations!".format(self.config.N_EARLY_STOPPING_NDROPS))
                    break

        self._print("End of Program")

        # OUTPUT REPORT:
        if gen_plot:
            report_.output_progress_plot(
                figsize       = (15,12),
                OUT_DIR       = self.folder_dict["analysis"],
                tag           = self.config.MODEL_TAG
            )
        # SAVE Model:
        self.save_model(tag="final")
        return report_
    
    def predict(self, pd_data=None, tag=None):
        if pd_data is None:
            pd_data_processed = self.final_testing_dataset
        else:
            raise ValueError("Feature Disabled, uncomment to predict other materials")
            # pre-process:
            # self._print("Pre-processing Test Dataset ...")
            # path_lite = os.path.join(self.folder_dict["processed_data"], 'preprocessed-lite-test-[{}].csv'.format(self.config.PRE_PROCESS_TAG))
            # path = os.path.join(self.folder_dict["processed_data"], 'preprocessed-idx-test-[{}].csv'.format(self.config.PRE_PROCESS_TAG))
            # if os.path.exists(path) and not self.config.FORCE_REBUILD:
            #     self._print("Loading Pre-Processed Test Dataset From {}".format(path))
            #     pd_data_processed = pd.read_csv(path)
            # else:
            #     pd_data_processed = self.generate_tweet_message_normalized_column(
            #         pd_data = pd_data, config = self.config
            #     )
            #     pd_data_processed.to_csv(path_lite)
            #     self._print("Pre-Processed Test Dataset (Lite) Saved @ {}".format(path_lite))
            #     pd_data_processed = self.generate_tweet_message_normalized_column_converted(
            #         pd_data = pd_data_processed
            #     )
            #     pd_data_processed.to_csv(path)
            #     self._print("Processed Test Dataset Saved @ {}".format(path))
        # prediction:
        self._print("Predicting ...")
        y_pred = []
        with torch.no_grad(): # Not training!
            for x in pd_data_processed["norm-tweet-bow-idx-array"]:
                if self.if_read_from_file:
                    bow_vec = self.convert_bow_idx_array_2_vector(literal_eval(x))
                else:
                    bow_vec = self.convert_bow_idx_array_2_vector(x)
                if self.config.USE_GPU:
                    bow_vec = bow_vec.to(self.device)
                log_probs = self.model(bow_vec)
                y_pred.append(log_probs.argmax(dim=1))
        # convert to df_pred
        self._print("Converting to dataframe ...")
        df_pred = pd.DataFrame({'label':[y.tolist()[0] for y in y_pred]})
        # save to file:
        if tag is not None:
            path = os.path.join(self.folder_dict["y-pred"], 'test_y_pred-[{}-{}].csv'.format(self.config.MODEL_TAG, tag))
            self._print("Prediction of the Test Dataset Saved @ {}".format(path))
            df_pred.to_csv(path, index_label="id")

        return df_pred

    def save_model(self, tag):
        torch.save(self.model.state_dict(), os.path.join(self.folder_dict["models"], "model-{}-{}.pt".format(self.config.MODEL_TAG, tag)))


# USER PARAMS: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# Pre-evaluation:
# CONFIG = TwitterLikePredictor.PredictorConfiguration(
#     Y_TAG                 = "likes_count",
#     USE_GPU               = True,
#     OUTPUT_FOLDER         = abspath("output"),
#     MODEL_TAG             = "test-v2.5-cross",
#     # tweet pre-processing:
#     DELIMITER_SET         = '; |, |、|。| ',
#     SYMBOLE_REMOVE_LIST   = ["\[", "\]", "\(", "\)"],
#     KEYS_TO_REMOVE_LIST   = ["http", "arXiv", "https"],
#     # training set:
#     SHUFFLE_TRAINING      = False,
#     PERCENT_TRAINING_SET  = 0.90, # 0.99
#     # bag of words (Tweeter Interpretation):
#     LOSS_FUNC             = nn.NLLLoss(),
#     BOW_TOTAL_NUM_EPOCHS  = 10, # 20
#     LEARNING_RATE         = 0.001,
#     FORCE_REBUILD         = True, 
#     OPTIMIZER             = optim.SGD,
#     MOMENTUM              = 0.8,
#     MODEL_VERSION         = "v2"
#     # PERCENT_TRAINING_SET  = 0.9,
#     # # Best Model so far: test-epoch-1 (Incorporate time ... info.) => 0.462 acc --> 0.45604
#     # BOW_TOTAL_NUM_EPOCHS  = 10,
#     # LEARNING_RATE         = 0.001,
#     # # Best Model so far: test-epoch-1 => 0.4365 acc --> 0.44792
#     # BOW_TOTAL_NUM_EPOCHS  = 20,
#     # LEARNING_RATE         = 0.001,
#     # # Best Model so far: test-epoch-180
#     # BOW_TOTAL_NUM_EPOCHS  = 180,
#     # LEARNING_RATE         = 0.001,
# )

# # INIT ENGINE: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# TLP_Engine = TwitterLikePredictor(pd_data_training=TRAIN_DATA, verbose=True, config=CONFIG)

# # TRAIN: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# report = TLP_Engine.train(gen_plot=True, sample_threshold=0.5)

# # PREDICTION: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# pd_data_processed, df_pred = TLP_Engine.predict(pd_data=TEST_DATA_X, tag="test")


# -------------------------------- MODEL AUTOMATION -------------------------------- %% #
"""
We will try to train possible models and choose the best model
Note: Local Validation is not representitive and deviates from test, so its a sole reference how well it might be.
"""
# # Auto overnight training: ----- ----- ----- ----- ----- ----- ----- -----
DICT_OF_CONFIG = {
    # WIP: 
    "dev-1-final-run3": TwitterLikePredictor.PredictorConfiguration(
        MODEL_TAG             = "latest-v1-emphasize-rebuild-3",
        PRE_PROCESS_TAG       = "latest-v1-emphasize-rebuild-3",
        BOW_TOTAL_NUM_EPOCHS  = 50,
        LEARNING_RATE         = 0.0005,
        PERCENT_TRAINING_SET  = None,
        SHUFFLE_TRAINING      = True,
        # FORCE_REBUILD         = True,
        MODEL_VERSION         = "v",
        N_EARLY_STOPPING_NDROPS = None, # no early stopping
    ), # [Best Best so far, 0.53057 on Kaggle  @ epoch 50 ] **************************** #
    # "dev-1-adam": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "latest-v1-emphasize-2",
    #     PRE_PROCESS_TAG       = "latest-v1-emphasize-2",
    #     BOW_TOTAL_NUM_EPOCHS  = 200,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.90,
    #     OPTIMIZER             = optim.Adam,
    #     MODEL_VERSION         = "v",
    #     N_EARLY_STOPPING_NDROPS = None, # no early stopping, full 80 iterations
    # ),
    # "dev-1-emphasize2": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "latest-v1-emphasize2",
    #     PRE_PROCESS_TAG       = "latest-v1-emphasize2",
    #     BOW_TOTAL_NUM_EPOCHS  = 200,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.99,
    #     MODEL_VERSION         = "v",
    #     FORCE_REBUILD         = True,
    #     N_EARLY_STOPPING_NDROPS = None, # no early stopping, full 80 iterations
    # ), 
    # "dev-1-emphasize3": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "latest-v1-emphasize3",
    #     PRE_PROCESS_TAG       = "latest-v1-emphasize3",
    #     BOW_TOTAL_NUM_EPOCHS  = 80,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.90,
    #     MODEL_VERSION         = "v"
    # ), 
    # "dev-1-emphasize1": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "latest-v1-emphasize",
    #     PRE_PROCESS_TAG       = "latest-v1-emphasize",
    #     BOW_TOTAL_NUM_EPOCHS  = 80,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.90,
    #     MODEL_VERSION         = "v"
    # ), # [Best Best so far, 0.521 on Kaggle  @ epoch 80 | TEST ACC: 0.7893] **************************** #
    #
    # |========== Retired Models:============================================================= |
    # "dev-1-emphasize3-all": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "latest-v1-emphasize3-all",
    #     PRE_PROCESS_TAG       = "latest-v1-emphasize3-all",
    #     BOW_TOTAL_NUM_EPOCHS  = 80,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.90,
    #     MODEL_VERSION         = "v"
    # ),  => collecting testing data features seems to overwhelm the model, might work better with variance sampling
    #= Comment:  (max:0.4969?) => 0.49163  [KING MODEL!] @ epoch:32
    # "dev-1-nn2-drop": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "min-variance-1-top10k",
    #     PRE_PROCESS_TAG       = "min-variance-1-top10k",
    #     BOW_TOTAL_NUM_EPOCHS  = 80,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.90,
    #     D_HIDDEN              = 4000,
    #     N_MIN_VARIANCE        = 1.0,
    #     N_TOP_FEATURES        = 10000,
    #     MODEL_VERSION         = "v2-drop"
    # ), 
    # "dev-1-nn2-2k": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "min-variance-top2k",
    #     PRE_PROCESS_TAG       = "min-variance-top2k",
    #     BOW_TOTAL_NUM_EPOCHS  = 80,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.90,
    #     D_HIDDEN              = 4000,
    #     N_TOP_FEATURES        = 2000,
    #     MODEL_VERSION         = "v2"
    # ), 
    # "dev-1-2k": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "min-variance-top2k",
    #     PRE_PROCESS_TAG       = "min-variance-top2k",
    #     BOW_TOTAL_NUM_EPOCHS  = 80,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.90,
    #     N_MIN_VARIANCE        = 1.0,
    #     N_TOP_FEATURES        = 2000,
    #     MODEL_VERSION         = "v"
    # ), 
    # "dev-1-nn2": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "min-variance-1-top10k",
    #     PRE_PROCESS_TAG       = "min-variance-1-top10k",
    #     BOW_TOTAL_NUM_EPOCHS  = 80,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.90,
    #     D_HIDDEN              = 4000,
    #     N_MIN_VARIANCE        = 1.0,
    #     N_TOP_FEATURES        = 10000,
    #     MODEL_VERSION         = "v2"
    # ), 
    # "dev-1-nn3": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "min-variance-1-top10k",
    #     PRE_PROCESS_TAG       = "min-variance-1-top10k",
    #     BOW_TOTAL_NUM_EPOCHS  = 80,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.90,
    #     D_HIDDEN              = 4000,
    #     N_MIN_VARIANCE        = 1.0,
    #     N_TOP_FEATURES        = 10000,
    #     MODEL_VERSION         = "v3"
    # ), 
    # "dev-1": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "min-variance-1-top10k",
    #     PRE_PROCESS_TAG       = "min-variance-1-top10k",
    #     BOW_TOTAL_NUM_EPOCHS  = 80,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.90,
    #     N_MIN_VARIANCE        = 1.0,
    #     N_TOP_FEATURES        = 10000,
    #     MODEL_VERSION         = "v"
    # ), 
    # "dev-2": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "min-variance-1-top5k",
    #     PRE_PROCESS_TAG       = "min-variance-1-top5k",
    #     BOW_TOTAL_NUM_EPOCHS  = 80,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.90,
    #     N_TOP_FEATURES        = 5000,
    #     MODEL_VERSION         = "v"
    # ), 
    #
    # "test-99pa-unify-emoji": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "top-10k-all-unify-emoji",
    #     PRE_PROCESS_TAG       = "top-10k-all-unify-emoji",
    #     BOW_TOTAL_NUM_EPOCHS  = 40,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.99,
    #     MODEL_VERSION         = "v",
    #     N_TOP_FEATURES        = 10000, # lets pick top 10k most appeared features
    #     ENABLE_EXTRA_CONVERSION = ["unify-emoji"],
    # ), #= Comment:   (max:?)
    # "test-nn-unify-emoji": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "top-10k-all-unify-emoji",
    #     PRE_PROCESS_TAG       = "top-10k-all-unify-emoji",
    #     BOW_TOTAL_NUM_EPOCHS  = 40,
    #     D_HIDDEN              = 2000,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.90,
    #     MODEL_VERSION         = "v2",
    #     N_TOP_FEATURES        = 10000, # lets pick top 10k most appeared features
    #     ENABLE_EXTRA_CONVERSION = ["unify-emoji"],
    # ), #= Comment:   (max:?)
    # "test-99pa": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "top-10k-all",
    #     PRE_PROCESS_TAG       = "top-10k-all",
    #     BOW_TOTAL_NUM_EPOCHS  = 40,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.99,
    #     MODEL_VERSION         = "v",
    #     N_TOP_FEATURES        = 10000 # lets pick top 10k most appeared features
    # ), #= Comment:   (max:?)
    # "test-nn": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "neural-network",
    #     PRE_PROCESS_TAG       = "top-10k-all",
    #     BOW_TOTAL_NUM_EPOCHS  = 80,
    #     D_HIDDEN              = 1000,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.99,
    #     MODEL_VERSION         = "v2",
    #     N_TOP_FEATURES        = 10000 # lets pick top 10k most appeared features
    # ), #= Comment:  (max:0.48805)
    # "test-1": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "top-10k-all",
    #     PRE_PROCESS_TAG       = "top-10k-all",
    #     BOW_TOTAL_NUM_EPOCHS  = 80,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.90,
    #     MODEL_VERSION         = "v",
    #     N_TOP_FEATURES        = 10000 # lets pick top 10k most appeared features
    # ), #= Comment:  (max:0.49687)
    # "test-3": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "no-tweet",
    #     PRE_PROCESS_TAG       = "no-tweet",
    #     BOW_TOTAL_NUM_EPOCHS  = 80,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.90,
    #     MODEL_VERSION         = "v"
    # ), #= Comment:  (max:0.4628) stopped, =. but proved that tweet is not significantly improving => we neeed PCA or reduction on features
    # "trial-2": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "3layer",
    #     BOW_TOTAL_NUM_EPOCHS  = 50,
    #     D_HIDDEN              = 4000,
    #     LEARNING_RATE         = 0.0001,
    #     MODEL_VERSION         = "v3"
    # ), => TODO: to be tested with deeper layer
    # "with-username": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "trial-1",
    #     BOW_TOTAL_NUM_EPOCHS  = 32,
    #     LEARNING_RATE         = 0.0001,
    #     PERCENT_TRAINING_SET  = 0.99,
    #     MODEL_VERSION         = "v"
    # ), #= Comment:  (max:0.4969?) => 0.49163  [KING MODEL!] @ epoch:32
    # "trial-1": TwitterLikePredictor.PredictorConfiguration(
    #     MODEL_TAG             = "trial-2",
    #     BOW_TOTAL_NUM_EPOCHS  = 30,
    #     LEARNING_RATE         = 0.0001,
    #     MODEL_VERSION         = "v"
    # ), #= Comment:  (max:0.4953?) (ep30: 0.4933 ==>  0.48829)
}

min_threshold = 0.51
THRESHOLD = min_threshold
for name_, config_ in DICT_OF_CONFIG.items():
    print("================================ ================================ BEGIN:{} , Goal:{} =>".format(name_, min_threshold))
    # INIT ENGINE: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    TLP_Engine = TwitterLikePredictor(
        pd_data_training=TRAIN_DATA,
        pd_data_testing=TEST_DATA_X,
        verbose=True, 
        verbose_show_sample_language_parse=False, 
        config=config_,
        engine_name=name_
    )

    # TRAIN: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    report = TLP_Engine.train(gen_plot=True, sample_threshold=THRESHOLD)
    max_validation_acc = np.max(report.history["test_acc"])
    if max_validation_acc > min_threshold:
        print("\n>>>> Best Model So Far: {} \n".format(max_validation_acc))
        min_threshold = max_validation_acc # rise standard

    # PREDICTION: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    df_pred = TLP_Engine.predict(tag="final")





# %%

# # %% -------------------------------- SECTION BREAK LINE -------------------------------- %% #
# """
# We will do post-analysis here, to see the validation performance of the model.
# """
# class TLP_Engine_Analyzer:
#     def __init__(self, TLP_Engine: TwitterLikePredictor):
#         self._Engine = TLP_Engine 
    
#     def reportBoW(self):
#         # # # Word Analysis: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
#         list_top100 = list(map(np.array, zip(* TLP_Engine.word_count_top_features)))
#         # Plot Language and video Count:
#         fig = plt.figure(figsize=(40,40))
#         ax = plt.subplot(1, 1, 1)
#         ax.set_title("Top 100 Repeated Word Count")
#         plt.bar(list_top100[0], list_top100[1][:, 0], label="0")
#         plt.bar(list_top100[0], list_top100[1][:, 1], label="1")
#         plt.bar(list_top100[0], list_top100[1][:, 2], label="2")
#         plt.bar(list_top100[0], list_top100[1][:, 3], label="3")


# %%
