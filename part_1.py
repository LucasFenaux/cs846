from transformers import AutoTokenizer, T5ForSequenceClassification
import pandas as pd
import torch
from torch.optim.adam import Adam
from tqdm import tqdm
import numpy as np


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base", max_length=4098, truncation=True)
model = T5ForSequenceClassification.from_pretrained("google-t5/t5-base", num_labels=5)

for param in model.transformer.parameters():
    param.requires_grad = False

for param in model.classification_head.parameters():
    param.requires_grad = True

model = model.to(device)

opt = Adam(model.classification_head.parameters(), lr=1e-5)
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['Prompt'] = 'What is the priority (from 0, highest priority, to 4, lowest priority) of the code bug given the following description. | Component: ' + train['Component'] + " | " + 'Title: ' + train['Title'] + " | " + 'Status: ' + train['Status'] + " | " + 'Resolution: ' + train['Resolution'] + " | " + 'Description: ' + train['Description']
x = train['Prompt'].to_numpy()
y = torch.LongTensor(train['Priority'].to_numpy())
# model.config.num_labels = 5
# labels = torch.LongTensor([0,1,2,3,4]).to(device)
# lengths = []
pbar = tqdm(zip(x,y), total=len(x))
for inp, lab in pbar:
    if isinstance(inp, str):
        inp_ids = tokenizer(inp, return_tensors="pt", truncation=True, max_length=4098).input_ids.to(device)
        # lengths.append(len(inp_ids.input_ids))
        outputs = model(input_ids=inp_ids, labels=lab)
        opt.zero_grad()
        outputs.loss.backward()
        opt.step()
# print(min(lengths), max(lengths), np.mean(lengths), np.std(lengths), sum([i < 4096 for i in lengths])/len(lengths))
# print(x)
# print(y)

#
# input_ids = tokenizer(
#     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
# ).input_ids  # Batch size 1
# outputs = model(input_ids=input_ids)
# last_hidden_states = outputs.last_hidden_state

