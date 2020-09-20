import json
import os
import logging
import random
from collections import namedtuple, OrderedDict, defaultdict
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from coref_bucket_batch_sampler import BucketBatchSampler
from data import get_dataset
from decoding import Decoder
from metrics import CorefEvaluator, MentionEvaluator
from utils import write_examples, EVAL_DATA_FILE_NAME, EvalDataPoint, extract_clusters, \
    extract_mentions_to_predicted_clusters_from_clusters, extract_clusters_for_decode

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, args, tokenizer, sampling_prob=1.0):
        self.args = args
        self.eval_output_dir = args.output_dir
        self.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        self.tokenizer = tokenizer
        self.sampling_prob = sampling_prob

    def evaluate(self, model, prefix=""):
        eval_dataset = get_dataset(self.args, tokenizer=self.tokenizer, evaluate=True)

        if self.eval_output_dir and not os.path.exists(self.eval_output_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(self.eval_output_dir)

        # Note that DistributedSampler samples randomly
        # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = BucketBatchSampler(eval_dataset, max_total_seq_len=self.args.max_total_seq_len)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Batch size = %d", self.eval_batch_size)
        logger.info("  Examples number: %d", len(eval_dataset))
        model.eval()

        post_pruning_mention_evaluator = MentionEvaluator()
        mention_evaluator = MentionEvaluator()
        coref_evaluator = CorefEvaluator()
        losses = defaultdict(list)
        for batch in eval_dataloader:
            if random.random() > self.sampling_prob:
                continue

            batch = tuple(tensor.to(self.args.device) for tensor in batch)
            input_ids, attention_mask, start_entity_mentions_indices, end_entity_mentions_indices, start_antecedents_indices, end_antecedents_indices, gold_clusters = batch

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                start_entity_mention_labels=start_entity_mentions_indices,
                                end_entity_mention_labels=end_entity_mentions_indices,
                                start_antecedent_labels=start_antecedents_indices,
                                end_antecedent_labels=end_antecedents_indices,
                                gold_clusters=gold_clusters,
                                return_all_outputs=True)
                loss_dict = outputs[-1]

            if self.args.n_gpu > 1:
                loss_dict = {key: val.mean() for key, val in loss_dict.items()}

            for key, val in loss_dict.items():
                losses[key].append(val.item())

            outputs = outputs[1:-1]

            batch_np = tuple(tensor.cpu().numpy() for tensor in batch)
            outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)
            for output in zip(*(batch_np + outputs_np)):
                gold_clusters = output[6]
                gold_clusters = extract_clusters(gold_clusters)
                mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
                gold_mentions = list(mention_to_gold_clusters.keys())

                if self.args.end_to_end:
                    starts, end_offsets, coref_logits, mention_logits = output[-4:]

                    max_antecedents = np.argmax(coref_logits, axis=1).tolist()
                    mention_to_antecedent = {((start, end), (starts[max_antecedent], end_offsets[max_antecedent])) for start, end, max_antecedent in
                                             zip(starts, end_offsets, max_antecedents) if max_antecedent < len(starts)}

                    predicted_clusters, _ = extract_clusters_for_decode(mention_to_antecedent)
                    candidate_mentions = list(zip(starts, end_offsets))
                elif self.args.baseline:
                    starts, end_offsets, coref_logits, mention_logits = output[-4:]

                    max_antecedents = np.argmax(coref_logits, axis=1).tolist()
                    mention_to_antecedent = {((start, start + end_offest), (starts[max_antecedent], starts[max_antecedent] + end_offsets[max_antecedent])) for start, end_offest, max_antecedent in
                                             zip(starts, end_offsets, max_antecedents) if max_antecedent < len(starts)}

                    predicted_clusters, _ = extract_clusters_for_decode(mention_to_antecedent)
                    candidate_mentions = list(zip(starts, starts + end_offsets))
                else:
                    data_point = EvalDataPoint(*output)

                    decoder = Decoder(use_mention_logits_for_antecedents=self.args.use_mention_logits_for_antecedents,
                                      use_mention_oracle=self.args.use_mention_oracle,
                                      gold_mentions=gold_mentions,
                                      gold_clusters=gold_clusters,
                                      use_crossing_mentions_pruning=self.args.use_crossing_mentions_pruning,
                                      only_top_k=self.args.only_top_k)

                    predicted_clusters, candidate_mentions = decoder.cluster_mentions(data_point.mention_logits,
                                                                                      data_point.start_coref_logits,
                                                                                      data_point.end_coref_logits,
                                                                                      data_point.attention_mask,
                                                                                      self.args.top_lambda)

                mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(predicted_clusters)
                predicted_mentions = list(mention_to_predicted_clusters.keys())
                post_pruning_mention_evaluator.update(candidate_mentions, gold_mentions)
                mention_evaluator.update(predicted_mentions, gold_mentions)
                coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted_clusters,
                                       mention_to_gold_clusters)

        post_pruning_mention_precision, post_pruning_mentions_recall, post_pruning_mention_f1 = post_pruning_mention_evaluator.get_prf()
        mention_precision, mentions_recall, mention_f1 = mention_evaluator.get_prf()
        prec, rec, f1 = coref_evaluator.get_prf()

        results = [(key, sum(val) / len(val)) for key, val in losses.items()]
        results += [
            ("post pruning mention precision", post_pruning_mention_precision),
            ("post pruning mention recall", post_pruning_mentions_recall),
            ("post pruning mention f1", post_pruning_mention_f1),
            ("mention precision", mention_precision),
            ("mention recall", mentions_recall),
            ("mention f1", mention_f1),
            ("precision", prec),
            ("recall", rec),
            ("f1", f1)
        ]
        logger.info("***** Eval results {} *****".format(prefix))
        for key, values in results:
            if isinstance(values, float):
                logger.info(f"  {key} = {values:.3f}")
            else:
                logger.info(f"  {key} = {values}")

        if self.eval_output_dir:
            output_eval_file = os.path.join(self.eval_output_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                if prefix:
                    writer.write(f'\n{prefix}:\n')
                for key, values in results:
                    if isinstance(values, float):
                        writer.write(f"{key} = {values:.3f}\n")
                    else:
                        writer.write(f"{key} = {values}\n")

        results = OrderedDict(results)
        results["experiment_name"] = self.args.experiment_name
        results["data"] = prefix
        with open(os.path.join(self.args.output_dir, "results.jsonl"), "a+") as f:
            f.write(json.dumps(results) + '\n')
        return results
