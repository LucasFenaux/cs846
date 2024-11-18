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

class TextDataset(Dataset):
    def __init__(self, x, y, tokenizer):
        self.tokenizer = tokenizer
        self.x = x
        self.y = y
        assert len(self.x) == len(self.y)

    def __getitem__(self, index):
        
        x = self.x[index]

        item = self.tokenizer(x, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
        item["input_ids"] =  item["input_ids"].squeeze(dim=0)
        item["attention_mask"] = item["attention_mask"].squeeze(dim=0)
        return item

    def __len__(self):
        return len(self.x)


def preprocess_dataset(tokenizer, train_ratio: float = 0.8, dataset_name='train.csv'):
    print("RUNNING " + dataset_name)
    train = pd.read_csv(dataset_name)
    train.fillna(" ",inplace=True)  # remove the nans
    train['Issue_id'] = train['Issue_id'].astype(int)
    train['Prompt'] = (('What is the priority (from 0, highest priority, to 4, lowest priority) of the code bug given the '
                       'following description. | Component: ') + train['Component'] + " | " + 'Title: ' + train['Title']
                       + " | " + 'Status: ' + train['Status'] + " | " + 'Resolution: ' + train['Resolution'] + " | " +
                       'Description: ' + train['Description'])
    
    # TODO why is this cast to numpy()?
    x = train['Prompt'].to_numpy()
    # dummy labels
    y = [np.array(1) for _ in range(len(train))]
  
    train_dataset = TextDataset(x, y,tokenizer)

    return train_dataset, train

def compute_metrics(output):
    logits = output.predictions[0]
    preds = np.argmax(logits, axis=-1)

    return preds

def main():
    dataset_file = "./shortened_test.csv"
    local_path = '/home/b3schnei/cs846_best_runs/cs846_output/checkpoint-6575'
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))[0]

    tokenizer = AutoTokenizer.from_pretrained(local_path, max_length=max_length, truncation=True)
    model = T5ForSequenceClassification.from_pretrained(local_path, num_labels=5)

    train, df  = preprocess_dataset(tokenizer, dataset_name=dataset_file, train_ratio=1)
    
    #TODO remove this
    train = torch.utils.data.Subset(train, list(range(100)))

    trainer = Trainer(model=model,
                    args=training_args,
                    train_dataset=train,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics
                    )
    out = trainer.predict(train)
    preds = compute_metrics(out)
    

    print("pred done")

if __name__ == '__main__':
    main()
