import torch
from torch.utils.data import Dataset
import pandas as pd


class PromptDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.train = pd.read_csv('./updated_train.csv')
        self.test = pd.read_csv('test.csv')
        self.apply_prompt_template()

    def __getitem__(self, index):
        x = self.train.iloc[index]['Prompt']
        y = self.train.iloc[index]['Mod_Prioirty']
        return x, y
    
    def __len__(self):
        return self.train.shape[0]

    def apply_prompt_template(self):
        self.train['Prompt'] = 'What is the priority of the code bug given the following description. Priority values can be 0, 1, 2, or 4. The highest priority ranking is 4 and the lowest priority ranking is 0. Please provide your reasoning and your answer in the following json object {bug_priority: <int>}\nBug Information:\nComponent: ' + self.train['Component'] + "\n" + 'Title: ' + self.train['Title'] + "\n" + 'Description: ' + self.train['Description'] + "\n" + 'Status: ' + self.train['Status'] + "\n" + 'Resolution: ' + self.train['Resolution'] + "\nGPT:"
        self.train['Generate_Prompt'] = 'Generate a code sample for which the following bug report applies. | Component: ' + self.train['Component'] + " | " + 'Title: ' + self.train['Title'] + " | " + 'Description: ' + self.train['Description'] + " | " + 'Status: ' + self.train['Status'] + " | " + 'Resolution: ' + self.train['Resolution']
