import torch.utils
from transformers import AutoTokenizer, T5ForSequenceClassification, Trainer, TrainingArguments, HfArgumentParser
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
import os
import sys
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
max_length = 256
#(torch25) b3schnei@growl:~/cs846$ python train_on_clean.py ./config/hf_config.json ./shortened_train.csv 

class TextDataset(Dataset):
    def __init__(self, x, y, tokenizer, return_dirty=False):
        self.tokenizer = tokenizer
        self.x = x
        self.y = y
        self.return_dirty = return_dirty
        assert len(self.x) == len(self.y)

    def __getitem__(self, index):
        
        x_clean, x_dirty = self.x[index]
        y = self.y[index]

        if self.return_dirty: x = x_dirty
        else: x = x_clean
        
        item = self.tokenizer(x, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
        item["input_ids"] =  item["input_ids"].squeeze(dim=0)
        item["attention_mask"] = item["attention_mask"].squeeze(dim=0)
        item["labels"] = y     
        return item

    def __len__(self):
        return len(self.x)

def preprocess_dataset(tokenizer, train_ratio: float = 0.8, dataset_name='train.csv'):
    print("RUNNING " + dataset_name)
    train = pd.read_csv(dataset_name)

    train.fillna('', inplace=True)  # remove the nans
    train['Prompt'] = (('What is the priority (from 0, highest priority, to 4, lowest priority) of the code bug given the '
                       'following description. | Component: ') + train['Component'] + " | " + 'Title: ' + train['Title']
                       + " | " + 'Status: ' + train['Status'] + " | " + 'Resolution: ' + train['Resolution'] + " | " +
                       'Description: ' + train['Shortened Description'])
    
    train['Dirty Prompt'] = (('What is the priority (from 0, highest priority, to 4, lowest priority) of the code bug given the '
                    'following description. | Component: ') + train['Component'] + " | " + 'Title: ' + train['Title']
                    + " | " + 'Status: ' + train['Status'] + " | " + 'Resolution: ' + train['Resolution'] + " | " +
                    'Description: ' + train['Description'])

    # TODO why is this cast to numpy()?
    x = train['Prompt'].to_numpy()
    x_dirty = train['Dirty Prompt'].to_numpy()
    y = train['Priority'].to_numpy(dtype=np.int32)
    classes = {}
    for inp, inp_dirty, lab in zip(x, x_dirty, y):
        item = {"clean":inp, "dirty":inp_dirty}
        
        if lab not in classes.keys():
            classes[lab] = [item]
        else:
            classes[lab].append(item)

    x_train = []
    y_train = []
    x_val = []
    y_val = []

    def unpack_to_tuple(l):
        for i in  range(len(l)):
            c = l[i]["clean"]
            d = l[i]["dirty"]
            l[i] = (c,d)
        return l

    for key in classes:
        r_train = np.random.choice(classes[key], int(train_ratio * len(classes[key])), replace=False)
        r_train = r_train.tolist()
        r_train_unpack = unpack_to_tuple(r_train)
        
        classes[key] = unpack_to_tuple(classes[key])

        r_val = list(set(classes[key]) - set(r_train_unpack))
        x_train += r_train_unpack
        x_val += r_val
        y_train += [key]*len(r_train_unpack)
        y_val += [key]*len(r_val)

    train_clean = TextDataset(x_train, y_train,tokenizer)
    val_clean = TextDataset(x_val, y_val, tokenizer)
    train_dirty = TextDataset(x_train, y_train,tokenizer, return_dirty=True)
    val_dirty = TextDataset(x_val, y_val, tokenizer, return_dirty=True)

    return train_clean, val_clean, train_dirty, val_dirty


def compute_metrics(output):
    labels = output.label_ids
    logits = output.predictions[0]
    preds = np.argmax(logits, axis=-1)

    accuracy = (preds==labels).mean()
    f1 = f1_score(labels, preds, average='macro')

    return {'val_acc': accuracy, "val_f1": f1}

def main():
    dataset_file = sys.argv[-1]
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large", max_length=max_length, truncation=True)
    model = T5ForSequenceClassification.from_pretrained("google-t5/t5-large", num_labels=5)
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-2]))[0]

    for param in model.parameters():
        param.requires_grad = True

    for param in model.classification_head.parameters():
        param.requires_grad = True

    _, v_c, t_d, v_d = preprocess_dataset(tokenizer, dataset_name=dataset_file)

    trainer = Trainer(model=model,
                    args=training_args,
                    train_dataset=t_d,
                    eval_dataset={
                        "clean_eval":v_c,
                        "dirty_eval":v_d
                    },
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics
                    )
    trainer.train()

if __name__ == '__main__':
    main()
