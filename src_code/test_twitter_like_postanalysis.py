"""
This main would analyze the give .csv
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
## Custom Files:
def abspath(relative_path):
    ABS_PATH = "/home/jx/JXProject/Github/UW__4B_Individual_Works/CS 480/Kaggle"
    return os.path.join(ABS_PATH, relative_path)

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(abspath("src_code"))

# Custom Lib
import jx_lib
from jx_pytorch_lib import ProgressReport
# %%
# model:
class BOW_Module(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BOW_Module, self).__init__()
        # the parameters of the affine mapping.
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        # input => Linear => softmax
        return F.log_softmax(self.linear(bow_vec), dim=1)

def convert_bow_idx_array_2_vector(bow_idx_array, VOCAB_SIZE):
    vec = torch.zeros(VOCAB_SIZE)
    for idx in bow_idx_array:
        vec[idx] += 1
    return vec.view(1, -1)

def make_target(label):
    return torch.LongTensor([label])


# from main_twitter_like import BOW_Module
# %% LOAD DATASET: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# path:
PATH_TRAIN_DATA_PROCESSED = abspath("processed_data/preprocessed-idx-train-[latest-v1-emphasize-rebuild].csv")
PATH_MODEL = abspath("output/dev-1-final-run2/models/model-latest-v1-emphasize-rebuild-final.pt")
PATH_ANAYSIS = abspath("output/dev-1-final-run2/analysis")
PATH_DICT = abspath("processed_data/bow-dict-[latest-v1-emphasize-rebuild].json")
# import data
TRAIN_DATA_PROCESSED = pd.read_csv(PATH_TRAIN_DATA_PROCESSED)
# import dictionary
with open(PATH_DICT, "r") as f:
    word_to_ix = json.load(f)

labels = TRAIN_DATA_PROCESSED["likes_count"].unique()
VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = len(labels)
# import model
model = BOW_Module(NUM_LABELS, VOCAB_SIZE)
model.load_state_dict(torch.load(PATH_MODEL))
model.eval()


# ## POST-ANALYSIS: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
n = len(TRAIN_DATA_PROCESSED)
validation_data = TRAIN_DATA_PROCESSED[int(n*0.9):n]
n = len(validation_data)
list_y_pred = []
list_right = []
print("Predicting ...")
i = 0
with torch.no_grad(): # Not training!
    for x, label in zip(validation_data["norm-tweet-bow-idx-array"], validation_data["likes_count"]):
        i += 1
        print("\r > Predicting [{}/{}]".format(i, n),  end='')

        bow_vec = convert_bow_idx_array_2_vector(literal_eval(x), VOCAB_SIZE)
        log_probs = model(bow_vec)
        y_pred = log_probs.argmax(dim=1).tolist()[0]

        list_y_pred.append(y_pred)
        list_right.append((y_pred == label))

validation_data["pred-likes"] = list_y_pred
validation_data["pred-ifcorrect"] = list_right

# ##  convert everything useful to quantity
validation_data["time-year"] = [0] * n
validation_data["time-month"] = [0] * n
validation_data["time-date"] = "" * n
validation_data["time-seconds"] = [0] * n
validation_data["time-zone"] = "" * n

validation_data["if-place"] = [False] * n
validation_data["if-quote"] = [False] * n
validation_data["if-thumbnail"] = [False] * n
validation_data["if-reply_to"] = [False] * n
validation_data["user_id_idx"] = [False] * n
validation_data["len-of-tweet"] = [False] * n
user_id_idx_dict = {}
for id_ in validation_data['id']:
    # convert creation time: => time affects how many ppl viewed the post
    time_str = validation_data.loc[id_, "created_at"]
    time_str = time_str.split(" ")
    date_ = datetime.datetime.strptime(time_str[0], "%Y-%m-%d")
    time_ = datetime.datetime.strptime(time_str[1], "%H:%M:%S")

    validation_data.loc[id_, "time-year"] = date_.year
    validation_data.loc[id_, "time-month"] = date_.month
    validation_data.loc[id_, "time-date"] = time_str[0]
    validation_data.loc[id_, "time-seconds"] = (time_ - datetime. datetime(1900, 1, 1)).total_seconds()
    validation_data.loc[id_, "time-zone"] = time_str[2]
    # other:
    validation_data.loc[id_, "if-place"] = not pd.isna(validation_data.loc[id_, "place"])
    validation_data.loc[id_, "if-quote"] = not pd.isna(validation_data.loc[id_, "quote_url"])
    validation_data.loc[id_, "if-thumbnail"] = not pd.isna(validation_data.loc[id_, "thumbnail"])
    validation_data.loc[id_, "if-reply_to"] = len(literal_eval(validation_data.loc[id_,"reply_to"])) 

    user_id = validation_data.loc[id_,"user_id"]
    if user_id not in user_id_idx_dict:
        user_id_idx_dict[user_id] = len(user_id_idx_dict)
    validation_data.loc[id_,"user_id_idx"] = user_id_idx_dict[user_id]

    validation_data.loc[id_, "len-of-tweet"] = len(validation_data.loc[id_,"tweet"])

validation_data.to_csv(os.path.join(PATH_ANAYSIS, "postanalysis.csv"))


# %% ## PLOTS: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
#  Confusion Matrix:
cf = confusion_matrix(validation_data["likes_count"], validation_data["pred-likes"])

fig, status = jx_lib.make_confusion_matrix(
    cf=cf,
    group_names=None,
    categories='auto',
    title="Prediction Summary"
)
fig.savefig("{}/plot_{}-conf_mat.png".format(PATH_ANAYSIS, "post-process-summary"), bbox_inches = 'tight')

# ## Entry Correlation Plot
partial_validation_data = pd.DataFrame(data=validation_data, 
    columns =["time-year", "time-month", "if-thumbnail", "likes_count", "pred-likes"])
corr = partial_validation_data.corr()
fig = plt.figure(figsize=(10,10))
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
fig.savefig("{}/plot_{}-correlation.png".format(PATH_ANAYSIS, "post-process-summary"), bbox_inches = 'tight')

# ## Result subplots
fig = plt.figure(figsize=(20,20))

DICT_SUBPLOTS = {
    "Prediction Result (likes_count)": {'x': "likes_count", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "Video vs. Correctness": {'x': "video", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "Time(s) vs. Correctness": {'x': "time-seconds", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "Time(zone) vs. Correctness": {'x': "time-zone", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "Time(year) vs. Correctness": {'x': "time-year", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "Time(month) vs. Correctness": {'x': "time-month", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "if-place vs. Correctness": {'x': "if-place", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "if-quote vs. Correctness": {'x': "if-quote", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "if-thumbnail vs. Correctness": {'x': "if-thumbnail", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "if-reply_to vs. Correctness": {'x': "if-reply_to", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "Userid vs. Correctness": {'x': "user_id_idx", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "Length-of-tweet vs. Correctness": {'x': "len-of-tweet", 'y':None, 'hue': "pred-ifcorrect", 'mult':"dodge"},
    "Userid vs. Likes": {'x': "user_id_idx", 'y':None, 'hue': "likes_count", 'mult':"dodge"},
    "#-of-reply vs. Likes": {'x': "if-reply_to", 'y':None, 'hue': "likes_count", 'mult':"dodge"},
    "Length-of-tweet vs. Likes": {'x': "len-of-tweet", 'y':None, 'hue': "likes_count", 'mult':"dodge"},
}
n_plot = np.ceil(np.sqrt(len(DICT_SUBPLOTS)))
i = 0
for title, entry in DICT_SUBPLOTS.items():
    i += 1
    ax = plt.subplot(n_plot, n_plot, i)
    ax.set_title(title)
    sns.histplot(ax=ax, data=validation_data, x=entry["x"], y=entry["y"], hue=entry["hue"], multiple=entry["mult"])

fig.savefig("{}/plot_{}.png".format(PATH_ANAYSIS, "post-process-summary"), bbox_inches = 'tight')



# %%
