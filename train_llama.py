import torch.utils
from transformers import AutoTokenizer, LlamaForSequenceClassification, Trainer, TrainingArguments, HfArgumentParser
import pandas as pd
import torch
from torch.optim.adam import Adam
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
import os
import sys
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
max_length = 256

class TextDataset(Dataset):
    def __init__(self, x, y, tokenizer):
        self.tokenizer = tokenizer
        self.x = x
        self.y = y
        assert len(self.x) == len(self.y)

    def __getitem__(self, index):
        
        x = self.x[index]
        y = self.y[index]

        item = self.tokenizer(x, return_tensors="pt", truncation=True, max_length=max_length)
        item["input_ids"] =  item["input_ids"].squeeze(dim=0)
        item["attention_mask"] = item["attention_mask"].squeeze(dim=0)
        item["labels"] = y     
        return item

    def __len__(self):
        return len(self.x)


def preprocess_dataset(tokenizer, train_ratio: float = 0.8):
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

    train_dataset = TextDataset(x_train, y_train,tokenizer)
    val_dataset = TextDataset(x_val, y_val, tokenizer)

    return train_dataset, val_dataset

def compute_metrics(output):
    labels = output.label_ids
    logits = output.predictions
    preds = np.argmax(logits, axis=-1)

    accuracy = (preds==labels).mean()
    f1 = f1_score(labels, preds, average='macro')

    return {'val_acc': accuracy, "val_f1": f1}

def main():
    # Change model path here
    model_name = "/home/b3schnei/pretrained/Llama-2-7b"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, max_length=max_length, truncation=True
    )
    # Set the padding token
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=5,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    # Update the model configuration
    model.config.pad_token_id = tokenizer.pad_token_id

    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))[0]

    for param in model.parameters():
        param.requires_grad = True

    for param in model.score.parameters():
        param.requires_grad = True

    train, val = preprocess_dataset(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

if __name__ == '__main__':
    main()
