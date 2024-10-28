import torch
import sys
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, GemmaTokenizer
from src.dataset import PromptDataset
from tqdm import tqdm
from typing import Union,Any, Mapping
device = "cuda" if torch.cuda.is_available() else "cpu"

# Courtesy of the folks @ HF
def _prepare_input(data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        return data.to(**kwargs)
    return data

model_to_test = sys.argv[2] if len(sys.argv) > 2 else "google/codegemma-7b-it"

if model_to_test == "google/codegemma-7b-it":
    #model = AutoModelForCausalLM.from_pretrained(model_to_test).to(device=device)
    tokenizer = GemmaTokenizer.from_pretrained(model_to_test, padding_side='left')

def collate_fn(batch):
    data = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    return data, labels 

dataset = PromptDataset()
dl = DataLoader(dataset, num_workers=1, batch_size=4, shuffle=True, collate_fn=collate_fn)
length = len(dataset)

def run_item(item):
    labels = item[1]
    inputs = item[0]
    #input_ids = tokenizer(inputs, return_tensors="pt", padding=True)
    #input_ids = _prepare_input(input_ids)
    print("\n")
    print(inputs[0])
    print("GROUND TRUTH:  " + str(labels[0]))
    print("\n")
    #outputs = model.generate(**input_ids, max_new_tokens=512)
    #generated_text = tokenizer.batch_decode(outputs)
    #for i, s in enumerate(generated_text): print(s.replace("<pad>", "") +"\n\n" + "GROUND TRUTH PRIORITY: " + str(labels[i]))

counter = 0

with torch.no_grad():
    for item in tqdm(dl,disable=True):
        run_item(item)
        if counter == 100: exit() 



