import json
import logging
import os
import pickle
from collections import namedtuple

import torch

from consts import SPEAKER_START, SPEAKER_END, NULL_ID_FOR_COREF
from utils import flatten_list_of_lists
from torch.utils.data import Dataset
import pandas as pd

CorefExample = namedtuple("CorefExample", ["token_ids", "clusters"])

logger = logging.getLogger(__name__)


class CorefDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_length=-1, is_generative=False):
        self.tokenizer = tokenizer
        logger.info(f"Reading dataset from {file_path}")
        examples, self.max_mention_num, self.max_cluster_size, self.max_num_clusters = self._parse_jsonlines(file_path)
        self.max_seq_length = max_seq_length
        self.examples, self.lengths, self.num_examples_filtered = self._tokenize(examples)
        self.is_generative = is_generative
        logger.info(
            f"Finished preprocessing Coref dataset. {len(self.examples)} examples were extracted, {self.num_examples_filtered} were filtered due to sequence length.")

    def _parse_jsonlines(self, file_path):
        examples = []
        max_mention_num = -1
        max_cluster_size = -1
        max_num_clusters = -1
        with open(file_path, 'r') as f:
            for line in f:
                d = json.loads(line.strip())
                doc_key = d["doc_key"]
                input_words = flatten_list_of_lists(d["sentences"])
                clusters = d["clusters"]
                max_mention_num = max(max_mention_num, len(flatten_list_of_lists(clusters)))
                max_cluster_size = max(max_cluster_size, max(len(cluster) for cluster in clusters) if clusters else 0)
                max_num_clusters = max(max_num_clusters, len(clusters) if clusters else 0)
                speakers = flatten_list_of_lists(d["speakers"])
                examples.append((doc_key, input_words, clusters, speakers))
        return examples, max_mention_num, max_cluster_size, max_num_clusters

    def _tokenize(self, examples):
        coref_examples = []
        lengths = []
        num_examples_filtered = 0
        for doc_key, words, clusters, speakers in examples:
            word_idx_to_start_token_idx = dict()
            word_idx_to_end_token_idx = dict()
            end_token_idx_to_word_idx = [0]  # for <s>

            token_ids = []
            last_speaker = None
            for idx, (word, speaker) in enumerate(zip(words, speakers)):
                if last_speaker != speaker:
                    speaker_prefix = [SPEAKER_START] + self.tokenizer.encode(" " + speaker,
                                                                             add_special_tokens=False) + [SPEAKER_END]
                    last_speaker = speaker
                else:
                    speaker_prefix = []
                for _ in range(len(speaker_prefix)):
                    end_token_idx_to_word_idx.append(idx)
                token_ids.extend(speaker_prefix)
                word_idx_to_start_token_idx[idx] = len(token_ids) + 1  # +1 for <s>
                tokenized = self.tokenizer.encode(" " + word, add_special_tokens=False)
                for _ in range(len(tokenized)):
                    end_token_idx_to_word_idx.append(idx)
                token_ids.extend(tokenized)
                word_idx_to_end_token_idx[idx] = len(token_ids)  # old_seq_len + 1 (for <s>) + len(tokenized_word) - 1 (we start counting from zero) = len(token_ids)

            if 0 < self.max_seq_length < len(token_ids):
                num_examples_filtered += 1
                continue

            new_clusters = [
                [(word_idx_to_start_token_idx[start], word_idx_to_end_token_idx[end]) for start, end in cluster] for
                cluster in clusters]
            lengths.append(len(token_ids))

            coref_examples.append(((doc_key, end_token_idx_to_word_idx), CorefExample(token_ids=token_ids, clusters=new_clusters)))
        return coref_examples, lengths, num_examples_filtered

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    def pad_clusters_inside(self, clusters):
        return [cluster + [(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)] * (self.max_cluster_size - len(cluster)) for cluster
                in clusters]

    def pad_clusters_outside(self, clusters):
        return clusters + [[]] * (self.max_num_clusters - len(clusters))

    def pad_clusters(self, clusters):
        clusters = self.pad_clusters_outside(clusters)
        clusters = self.pad_clusters_inside(clusters)
        return clusters

    def pad_batch(self, batch, max_length):
        max_length += 2  # we have additional two special tokens <s>, </s>
        padded_batch = []
        for example in batch:
            encoded_dict = self.tokenizer.encode_plus(example[0],
                                                      add_special_tokens=True,
                                                      pad_to_max_length=True,
                                                      max_length=max_length,
                                                      return_attention_mask=True,
                                                      return_tensors='pt')
            clusters = self.pad_clusters(example.clusters)
            example = (encoded_dict["input_ids"], encoded_dict["attention_mask"]) + (torch.tensor(clusters),)
            padded_batch.append(example)
        tensored_batch = tuple(torch.stack([example[i].squeeze() for example in padded_batch], dim=0) for i in range(len(example)))
        return tensored_batch


def get_dataset(args, tokenizer, evaluate=False):
    # TODO: add cache support
    if args.is_generative:
        val_dataset = pd.read_parquet(args.pandas_dataframe)
        inp = val_dataset.iloc[0]["input"]
        if not inp.startswith("coref: "):
            val_dataset["input"] = "coref: " + val_dataset["input"]
        val_dataset["input"]
        if args.num_examples:
            val_dataset = val_dataset[:args.num_examples]
        val_set = CorefPandasDataset(
            val_dataset,
            tokenizer,
            args.max_seq_length,
            args.max_seq_length,
            "input",
            "output",
        )
        return val_set

    read_from_cache, file_path = False, ''
    if evaluate and os.path.exists(args.predict_file_cache):
        file_path = args.predict_file_cache
        read_from_cache = True
    elif (not evaluate) and os.path.exists(args.train_file_cache):
        file_path = args.train_file_cache
        read_from_cache = True

    if read_from_cache:
        logger.info(f"Reading dataset from {file_path}")
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    file_path, cache_path = (args.predict_file, args.predict_file_cache) if evaluate else (args.train_file, args.train_file_cache)

    if args.is_generative:
        val_dataset = pd.read_parquet(file_path)
        val_set = CorefPandasDataset(
            val_dataset,
            tokenizer,
            args.max_seq_length,
            args.max_seq_length,
            "input",
            "output",
        )
    else:
        coref_dataset = CorefDataset(file_path, tokenizer, max_seq_length=args.max_seq_length)
    with open(cache_path, 'wb') as f:
        pickle.dump(coref_dataset, f)

    return coref_dataset


class CorefPandasDataset(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }
