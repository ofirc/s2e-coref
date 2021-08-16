#!/usr/bin/env python
#
# For more information:
# HuggingFace Course > Fine-tuning a pretrained model >
#   Fine-tuning a pretrained model with the Trainer API
# https://huggingface.co/course/chapter3/3?fw=pt
#
# Weights & Biases
# https://docs.wandb.ai/guides/integrations/huggingface
#

# On GoogleColab
# !pip install torch torchvision torchaudio
# !pip install transformers[sentencepiece]
# !pip install datasets
# !pip install wandb

# %env WANDB_PROJECT=nlp_coref

# On a VM:
# conda create --name nlp python
# conda activate nlp
# conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

#
# conda install sentencepiece
# conda install -c huggingface transformers
#

# pip install transformers[sentencepiece]
# pip install wandb datasets

# If you are running it for the first time you'll have to
# run it from command line instead of PyCharm and create an account or login:
# (hugging) C:\nlp\tmp>huggingface_trainer.py
# wandb: (1) Create a W&B account
# wandb: (2) Use an existing W&B account
# wandb: (3) Don't visualize my results
# wandb: Enter your choice:
#
#
#
import os
import torch
import numpy as np
import pandas as pd

from datasets import load_dataset
from datasets import load_metric
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import Trainer
from transformers import T5ForConditionalGeneration

if os.environ.get('NLP_USE_20'):
    TRAIN_FILE = 'train_with_paragraphs_20.parq'
    DEV_FILE = 'dev_with_paragraphs_20.parq'
    TEST_FILE = 'test_with_paragraphs_20.parq'
else:
    TRAIN_FILE = 'train_with_paragraphs.parq'
    DEV_FILE = 'dev_with_paragraphs.parq'
    TEST_FILE = 'test_with_paragraphs.parq'

for name, path in zip(('Train', 'Validation'), (TRAIN_FILE, DEV_FILE)):
    print(f'Using {path} for {name}')

#checkpoint = "google/t5-v1_1-base"
checkpoint = "./t5_model-google"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print(f'Tokenizer: {type(tokenizer).__name__}')

class CorefDataSet(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]
        # Uncomment to use pre-tokenized inputs.
        #self.my_tokenize()

    def my_tokenize(self):
        self.entries = []
        for index, row in self.data.iterrows():
            row_src = row['input']
            row_dst = row['output']
            source_text = str(row_src)
            target_text = str(row_dst)
            src = self.tokenizer.batch_encode_plus([source_text], return_tensors="pt", padding="max_length")
            target = self.tokenizer.batch_encode_plus([target_text], return_tensors="pt", padding="max_length")
            source_ids = src["input_ids"].squeeze()
            source_mask = src["attention_mask"].squeeze()
            target_ids = target["input_ids"].squeeze()
            target_mask = target["attention_mask"].squeeze()
            d = {
                "input_ids": source_ids.to(dtype=torch.long),
                "attention_mask": source_mask.to(dtype=torch.long),
                "decoder_input_ids": target_ids.to(dtype=torch.long),
                "labels": target_ids.to(dtype=torch.long),
            }
            self.entries.append(d)

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        # Uncomment to use pre-tokenized inputs.
        #return self.entries[index]

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        src = tokenizer.batch_encode_plus([source_text], return_tensors="pt",
                                          padding="max_length", truncation=True)
        target = tokenizer.batch_encode_plus([target_text], return_tensors="pt",
                                             padding="max_length", truncation=True)

        source_ids = src["input_ids"].squeeze()
        source_mask = src["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "input_ids": source_ids.to(dtype=torch.long),
            "attention_mask": source_mask.to(dtype=torch.long),
            "decoder_input_ids": target_ids.to(dtype=torch.long),
            "labels": target_ids.to(dtype=torch.long),
        }

# TODO: see if we can use it somehow
#model = AutoModel.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to('cuda')
model.eval()
print(f'Model: {type(model).__name__}')
#test_df = pd.read_parquet(TEST_FILE)
test_df = pd.read_parquet(TRAIN_FILE)

test_trainset = CorefDataSet(test_df, tokenizer, "input", "output")

NUM_EXAMPLES = 2

with torch.no_grad():
  for i in range(NUM_EXAMPLES):
    data = test_trainset[i]
    ids = data['input_ids'].to('cuda', dtype=torch.long).unsqueeze(0)
    mask = data['attention_mask'].to('cuda', dtype=torch.long).unsqueeze(0)
    generated_ids = model.generate(input_ids=ids, attention_mask=mask, max_length=512, num_beams=1)
    import pdb; pdb.set_trace()
    y = data['decoder_input_ids'].to('cuda', dtype=torch.long).unsqueeze(0)

pass
