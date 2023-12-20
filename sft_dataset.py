
import torch
import json
import random

from tqdm import tqdm
from torch.utils.data import Dataset

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super(SFTDataset, self).__init__()
        data = []
        print(f"Reading in data from file: {data_path}")
        with open(data_path, "r") as file:
            for line in file:  
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    print("json processing exception", e)
                    continue

        print(f"Got {len(data)} examples, preprocess...")
        data_dict = self.preprocess(data, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    
    def preprocess(self, examples, tokenizer):
        """
        Preprocess the data by tokenizing.
        """
        all_input_ids = []

        print("Tokenizing dataset...")
        for ex in tqdm(examples):
            # Add a positive example
            text = f"{ex['context']}\n\nQ: {ex['prompt']}\nA: {ex['response']}\n"
            tokenized = tokenizer.encode(text)
            all_input_ids.append(torch.LongTensor(tokenized))
            
            # Generate a negative example
            random_ex = random.choice(examples)
            text = f"{random_ex['context']}\n\nQ: {ex['prompt']}\nA: I don't know.\n"
            tokenized = tokenizer.encode(text)
            all_input_ids.append(torch.LongTensor(tokenized))

        random.shuffle(all_input_ids)

        return dict(input_ids=all_input_ids, labels=all_input_ids)