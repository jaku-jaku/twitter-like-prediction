"""
This main would analyze the give .csv
"""
# %% Import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

from dataclasses import dataclass
from typing import Dict, Any, List
import re
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

# %% LOAD DATASET: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
# import data
TRAIN_DATA_X = pd.read_csv(abspath("data/p_train_x.csv"))
TRAIN_DATA_Y = pd.read_csv(abspath("data/p_train_y.csv"))
TEST_DATA_X = pd.read_csv(abspath("data/p_test_x.csv"))
ic(np.sum(TRAIN_DATA_X["id"] == TRAIN_DATA_Y["id"])) # so the assumption should be right, they share the exact same id in sequence
TRAIN_DATA = pd.concat([TRAIN_DATA_X, TRAIN_DATA_Y["likes_count"]], axis=1)
ic(TRAIN_DATA.shape)

# %% Pre-Data Analysis: ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- #
ANALYSIS_OUTPUT_FOLDER =  abspath("output-analysis")
jx_lib.create_folder(ANALYSIS_OUTPUT_FOLDER)

TRAIN_DATA.head(10)
sns.displot(TRAIN_DATA["likes_count"])
HEADERS = list(TRAIN_DATA.columns)
a = ic(HEADERS)

# Plot Language and video Count:
fig = plt.figure(figsize=(20,20))
ax = plt.subplot(2, 2, 1)
ax.set_title("Language Count")
sns.histplot(ax=ax, data=TRAIN_DATA, x="language", hue="likes_count", multiple="dodge")
fig.savefig("{}/plot_{}.png".format(ANALYSIS_OUTPUT_FOLDER, "preprocess"), bbox_inches = 'tight')

ax = plt.subplot(2, 2, 2)
ax.set_title("Video Count")
sns.histplot(ax=ax, data=TRAIN_DATA, x="video", hue="likes_count", multiple="dodge")
fig.savefig("{}/plot_{}.png".format(ANALYSIS_OUTPUT_FOLDER, "preprocess"), bbox_inches = 'tight')


# %% Expand information & generate correlation map:
n = len(TRAIN_DATA)
TRAIN_DATA["time-year"] = [0] * n
TRAIN_DATA["time-month"] = [0] * n
TRAIN_DATA["time-date"] = "" * n
TRAIN_DATA["time-seconds"] = [0] * n
TRAIN_DATA["time-zone"] = "" * n

TRAIN_DATA["if-place"] = [False] * n
TRAIN_DATA["if-quote"] = [False] * n
TRAIN_DATA["if-thumbnail"] = [False] * n
TRAIN_DATA["if-reply_to"] = [False] * n
for i in TRAIN_DATA['id']:
    # convert creation time: => time affects how many ppl viewed the post
    time_str = TRAIN_DATA["created_at"][i].split(" ")
    date_ = datetime.datetime.strptime(time_str[0], "%Y-%m-%d")
    time_ = datetime.datetime.strptime(time_str[1], "%H:%M:%S")

    TRAIN_DATA["time-year"][i] = date_.year
    TRAIN_DATA["time-month"][i] = date_.month
    TRAIN_DATA["time-date"][i] = time_str[0]
    TRAIN_DATA["time-seconds"][i] = (time_ - datetime. datetime(1900, 1, 1)).total_seconds()
    TRAIN_DATA["time-zone"][i] = time_str[2]
    # other:
    TRAIN_DATA["if-place"][i] = pd.notnull(TRAIN_DATA["place"][i])
    TRAIN_DATA["if-quote"][i] = pd.notnull(TRAIN_DATA["quote_url"][i])
    TRAIN_DATA["if-thumbnail"][i] = pd.notnull(TRAIN_DATA["thumbnail"][i])
    TRAIN_DATA["if-reply_to"][i] = len(TRAIN_DATA["reply_to"][i]) > 0

validation_data.to_csv(abspath("output-analysis/training_data_expanded"))

# %%
corr1 = TRAIN_DATA.corr()
fig = plt.figure(figsize=(10,10))
ax = sns.heatmap(
    corr1, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)