import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 5)

from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import AdaBoostClassifier


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# train.head()
#
# train.describe()
#
# print("NaNs in train file", np.sum(np.sum(train.isna(), axis=0)) )
# print("NaNs in test file", np.sum(np.sum(test.isna(), axis=0)) )
# print(train.isna())
# print(test.isna())
# le = LabelEncoder()
# train['Status_num'] = le.fit_transform(train['Status'])
# train['Resolution_num'] = le.fit_transform(train['Resolution'])
#
# train.head()
#
# train.drop('Status', axis=1, inplace=True)
# train.drop('Resolution', axis=1, inplace=True)
#
# train.head()
#
# y=train['Priority']
# unique, counts = np.unique(y, return_counts=True)
# print("Classes:", unique.tolist())
# print("Counts:", counts.tolist())
#
# plt.bar(unique, counts, color=['g', 'orange', 'r'], alpha=0.7)
# plt.title("#Bugs VS Occurrences")
# plt.xticks(range(len(unique)))
# plt.ylabel("Occurrences")
# plt.xlabel("# Bugs")
# plt.show()
pd.set_option('display.max_columns', None)

train['Prompt'] = 'What is the priority (from 0, highest priority, to 4, lowest priority) of the code bug given the following description. | Component: ' + train['Component'] + " | " + 'Title: ' + train['Title'] + " | " + 'Description: ' + train['Description'] + " | " + 'Status: ' + train['Status'] + " | " + 'Resolution: ' + train['Resolution']
train['Generate_Prompt'] = 'Generate a code sample for which the following bug report applies. | Component: ' + train['Component'] + " | " + 'Title: ' + train['Title'] + " | " + 'Description: ' + train['Description'] + " | " + 'Status: ' + train['Status'] + " | " + 'Resolution: ' + train['Resolution']

train.head()
print(train.iloc[0]['Prompt'])

from transformers import GemmaTokenizer, AutoModelForCausalLM
from access_token import access_token
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GemmaTokenizer.from_pretrained("google/codegemma-7b-it")

model = AutoModelForCausalLM.from_pretrained("google/codegemma-7b-it").to(device)

input_text = train.iloc[0]['Generate_Prompt']
input_ids = tokenizer(input_text, return_tensors="pt").to(device)

outputs = model.generate(**input_ids, max_new_tokens=2048)
print(tokenizer.decode(outputs[0]))

input_text = train.iloc[0]['Prompt'] + "| Here is a possible code sample for the bug | " + tokenizer.decode(outputs[0])
input_ids = tokenizer(input_text, return_tensors="pt").to(device)

outputs = model.generate(**input_ids, max_new_tokens=2048)
print(tokenizer.decode(outputs[0]))
