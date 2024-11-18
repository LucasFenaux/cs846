import torch.utils
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, HfArgumentParser
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
from transformers import GenerationConfig
from src.utils.smoothed_value import SmoothedValue

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
max_length = 64


def main_vllm():

    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123"
    )
    mode = "train"
    train = pd.read_csv(mode + '.csv')
    # train.dropna(inplace=True)  # remove the nans
    train.fillna('', inplace=True)

    train['Prompt'] = (
        "Shorten the following code bug text description while preserving information relevant to its severity/priority. "
        "Only output the shortened text description and nothing else. Below is the description: \n"
        + train['Description']
    )
    train['Shortened Description'] = train['Description']
                # ('What is the priority (from 0, highest priority, to 4, lowest priority) of the code bug given the '
                #  'following description. | Component: ') + train['Component'] + " | " + 'Title: ' + train['Title']
                # + " | " + 'Status: ' + train['Status'] + " | " + 'Resolution: ' + train['Resolution'] + " | " +
                # 'Description: ' + train['Description'])
    new_train = train.copy()

    inputs = tqdm(train['Prompt'].to_numpy())
    shortened = SmoothedValue()
    old_len = SmoothedValue()
    new_len = SmoothedValue()
    for i, x in enumerate(inputs):
        if len(x) > 4096:
            x = x[:4096]
        completion = client.chat.completions.create(
            model="neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16",
            messages=[
                {"role": "user", "content": x}
            ]
        )
        # print("#################################################################")
        # print(x)
        # print("-----------------------------------------------------------------")
        new_desc = completion.choices[0].message.content
        # print(new_desc)
        shortened.update(len(x) - len(new_desc))
        old_len.update(len(x))
        new_len.update(len(new_desc))
        # print(len(new_desc))
        # new_desc = new_desc.replace(x, '')
        inputs.set_description(f"{i+1} | Shortened: {shortened.global_avg:.2f} | Old len: {old_len.global_avg:.2f} | New len {new_len.global_avg:.2f}")
        # print(f"{i} | Shortened: {len(x) - len(new_desc)} | Old len: {len(x)} | New len {len(new_desc)}")
        # print(new_desc)
        # print('########################################################################')
        # new_train.iloc[i]['Description'] = new_desc
        new_train.at[i, 'Shortened Description'] = new_desc


    new_train.to_csv(f'shortened_{mode}.csv', index=False)

if __name__ == '__main__':
    main_vllm()