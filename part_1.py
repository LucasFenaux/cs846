from sympy.physics.mechanics.tests.test_system import out_eqns
from transformers import AutoTokenizer, T5ForSequenceClassification
import pandas as pd
import torch
from torch.optim.adam import Adam
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from src.utils.smoothed_value import SmoothedValue
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
max_length = 1024
train_batch_size = 64
val_batch_size = 64

class TextDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert len(self.x) == len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def preprocess_dataset(train_ratio: float = 0.8):
    train = pd.read_csv('train.csv')
    train.dropna(inplace=True)  # remove the nans
    train['Prompt'] = (('What is the priority (from 0, highest priority, to 4, lowest priority) of the code bug given the '
                       'following description. | Component: ') + train['Component'] + " | " + 'Title: ' + train['Title']
                       + " | " + 'Status: ' + train['Status'] + " | " + 'Resolution: ' + train['Resolution'] + " | " +
                       'Description: ' + train['Description'])
    x = train['Prompt'].to_numpy()
    y = train['Priority'].to_numpy()
    classes = {}
    for inp, lab in zip(x, y):
        if lab not in classes.keys():
            classes[lab] = [inp]
        else:
            classes[lab].append(inp)

    x_train = []
    y_train = []
    x_val = []
    y_val = []
    for key in classes:
        r_train = np.random.choice(classes[key], int(train_ratio * len(classes[key])), replace=False).tolist()

        r_val = list(set(classes[key]) - set(r_train))
        x_train += r_train
        x_val += r_val
        y_train += [key]*len(r_train)
        y_val += [key]*len(r_val)

    train_dataset = TextDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=16)

    val_dataset = TextDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,  num_workers=16)
    return train_loader, val_loader

def test(dataloader, tokenizer, model):
    with torch.no_grad():
        loss = SmoothedValue()
        preds = []
        labels = []
        for x, y in dataloader:
            inp_ids = tokenizer(x, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length").input_ids.to(device)
            outputs = model(input_ids=inp_ids, labels=y)
            loss.update(outputs.loss.item())
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().detach().tolist())
            labels.extend(y.cpu().detach().tolist())
        f1 = f1_score(labels, preds, average='macro')
        print(f"Val Loss: {loss.global_avg:.3f} | Val F1 Score: {f1:.3f}")
        return loss.global_avg, f1

def train_one_epoch(dataloader, tokenizer, model, opt, epoch):
    pbar = tqdm(dataloader)
    loss = SmoothedValue()
    preds = []
    labels = []
    for x, y in pbar:
        inp_ids = tokenizer(x, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length").input_ids.to(device)
        outputs = model(input_ids=inp_ids, labels=y)
        opt.zero_grad()
        outputs.loss.backward()
        opt.step()
        loss.update(outputs.loss.item())
        preds.extend(torch.argmax(outputs.logits, dim=1).cpu().detach().tolist())
        labels.extend(y.cpu().detach().tolist())
        pbar.set_description(f"Epoch: {epoch} | Loss: {loss:.3f}")
    f1 = f1_score(labels, preds, average='macro')
    print(f"Train F1 Score: {f1:.3f}")
    return loss.global_avg, f1

def main():
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base", max_length=max_length, truncation=True)
    model = T5ForSequenceClassification.from_pretrained("google-t5/t5-base", num_labels=5)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classification_head.parameters():
        param.requires_grad = True

    model = model.to(device)

    opt = Adam(model.classification_head.parameters(), lr=1e-5)
    num_epochs = 5
    train_loader, val_loader = preprocess_dataset()
    f1_scores = []
    for epoch in range(num_epochs):
        train_one_epoch(train_loader, tokenizer, model, opt, epoch)
        _, score = test(val_loader, tokenizer, model)
        f1_scores.append(float(f"{score:.3f}"))

    print(f"Validation f1 scores through the epochs: {f1_scores}")

    # pbar = tqdm(zip(x,y), total=len(x))
    # for inp, lab in pbar:
    #     if isinstance(inp, str):
    #         inp_ids = tokenizer(inp, return_tensors="pt", truncation=True, max_length=4098).input_ids.to(device)
    #         # lengths.append(len(inp_ids.input_ids))
    #         outputs = model(input_ids=inp_ids, labels=lab)
    #         opt.zero_grad()
    #         outputs.loss.backward()
    #         opt.step()
# print(min(lengths), max(lengths), np.mean(lengths), np.std(lengths), sum([i < 4096 for i in lengths])/len(lengths))
# print(x)
# print(y)

#
# input_ids = tokenizer(
#     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
# ).input_ids  # Batch size 1
# outputs = model(input_ids=input_ids)
# last_hidden_states = outputs.last_hidden_state

if __name__ == '__main__':
    main()
