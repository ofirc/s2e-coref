import json
import os
import logging
import random
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from coref_bucket_batch_sampler import BucketBatchSampler
from data import get_dataset
from metrics import CorefEvaluator, MentionEvaluator
from utils import extract_clusters, extract_mentions_to_predicted_clusters_from_clusters, extract_clusters_for_decode
from conll import evaluate_conll

logger = logging.getLogger(__name__)

from collections import defaultdict
import itertools


def get_ranges(i):
    for a, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
        b = list(b)
        yield b[0][1], b[-1][1]


def convert_seq_to_spans(seq):
    my_dict = {i: key for i, key in enumerate(seq.split()) if key != '-'}
    v = defaultdict(list)
    for key, value in sorted(my_dict.items()):
        v[value].append(key)
    spans = []
    for value in v.values():
        tuples_to_add = tuple(get_ranges(value))
        spans.append(tuples_to_add)
    return spans


class Evaluator:
    def __init__(self, args, tokenizer):
        self.args = args
        self.eval_output_dir = args.output_dir
        self.tokenizer = tokenizer

    def evaluate(self, model, prefix="", tb_writer=None, global_step=None, official=False):
        eval_dataset = get_dataset(self.args, tokenizer=self.tokenizer, evaluate=True)

        if self.args.num_examples:
            eval_dataset.examples = eval_dataset.examples[:self.args.num_examples]
            eval_dataset.lengths = eval_dataset.lengths[:self.args.num_examples]

        if self.eval_output_dir and not os.path.exists(self.eval_output_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(self.eval_output_dir)

        # Note that DistributedSampler samples randomly
        # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = BucketBatchSampler(eval_dataset, max_total_seq_len=self.args.max_total_seq_len, batch_size_1=True)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Examples number: %d", len(eval_dataset))
        model.eval()

        post_pruning_mention_evaluator = MentionEvaluator()
        mention_evaluator = MentionEvaluator()
        coref_evaluator = CorefEvaluator()
        losses = defaultdict(list)
        doc_to_prediction = {}
        doc_to_subtoken_map = {}
        for (doc_key, subtoken_maps), batch in eval_dataloader:

            batch = tuple(tensor.to(self.args.device) for tensor in batch)
            input_ids, attention_mask, gold_clusters = batch

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
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
                gold_clusters = output[2]
                gold_clusters = extract_clusters(gold_clusters)
                mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
                gold_mentions = list(mention_to_gold_clusters.keys())

                starts, end_offsets, coref_logits, mention_logits = output[-4:]

                max_antecedents = np.argmax(coref_logits, axis=1).tolist()
                mention_to_antecedent = {((int(start), int(end)), (int(starts[max_antecedent]), int(end_offsets[max_antecedent]))) for start, end, max_antecedent in
                                         zip(starts, end_offsets, max_antecedents) if max_antecedent < len(starts)}

                predicted_clusters, _ = extract_clusters_for_decode(mention_to_antecedent)
                candidate_mentions = list(zip(starts, end_offsets))

                mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(predicted_clusters)
                predicted_mentions = list(mention_to_predicted_clusters.keys())
                post_pruning_mention_evaluator.update(candidate_mentions, gold_mentions)
                mention_evaluator.update(predicted_mentions, gold_mentions)
                coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted_clusters,
                                       mention_to_gold_clusters)
                doc_to_prediction[doc_key] = predicted_clusters
                doc_to_subtoken_map[doc_key] = subtoken_maps

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
            if tb_writer is not None and global_step is not None:
                tb_writer.add_scalar(key, values, global_step)

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

        if official:
            with open(os.path.join(self.args.output_dir, "preds.jsonl"), "w") as f:
                f.write(json.dumps(doc_to_prediction) + '\n')
                f.write(json.dumps(doc_to_subtoken_map) + '\n')

            if self.args.conll_path_for_eval is not None:
                conll_results = evaluate_conll(self.args.conll_path_for_eval, doc_to_prediction, doc_to_subtoken_map)
                official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
                logger.info('Official avg F1: %.4f' % official_f1)

        return results

class GenerativeEvaluator(Evaluator):
    def __init__(self, args, tokenizer):
        Evaluator.__init__(self, args, tokenizer)

    def evaluate(self, model, prefix="", tb_writer=None, global_step=None, official=False):
        eval_dataset = get_dataset(self.args, tokenizer=self.tokenizer, evaluate=True)

        # if self.args.num_examples:
        #     eval_dataset.examples = eval_dataset.examples[:self.args.num_examples]
        #     eval_dataset.lengths = eval_dataset.lengths[:self.args.num_examples]

        if self.eval_output_dir and not os.path.exists(self.eval_output_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(self.eval_output_dir)

        # Note that DistributedSampler samples randomly
        # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        #eval_dataloader = BucketBatchSampler(eval_dataset, max_total_seq_len=self.args.max_total_seq_len, batch_size_1=True)

        val_params = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 0,
        }

        # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
        from torch.utils.data import DataLoader
        val_loader = DataLoader(eval_dataset, **val_params)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Examples number: %d", len(eval_dataset))
        model.eval()

        predictions = []
        actuals = []
        coref_evaluator = CorefEvaluator()
        with torch.no_grad():
            for _, data in enumerate(val_loader, 0):
                y = data['target_ids'].to(self.args.device, dtype=torch.long)
                ids = data['source_ids'].to(self.args.device, dtype=torch.long)
                mask = data['source_mask'].to(self.args.device, dtype=torch.long)

                # TODO: try to fine-tune:
                #       length_penalty
                #       repetition_penalty - for longer sequences
                #       num beams
                #
                generated_ids = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_length=self.args.max_seq_length,
                    num_beams=self.args.num_beams,
                )
                preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                         generated_ids]
                target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
                if _ % 10 == 0:
                    logger.info(f'Completed {_}')

                preds_spans = convert_seq_to_spans(" ".join(preds))
                target_spans = convert_seq_to_spans(" ".join(target))

                mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(target_spans)
                mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(preds_spans)
                coref_evaluator.update(preds_spans, target_spans,
                                       mention_to_predicted_clusters,
                                       mention_to_gold_clusters)


                predictions.extend(preds)
                actuals.extend(target)

        prec, rec, f1 = coref_evaluator.get_prf()

        results = []
        results += [
            ("precision", prec),
            ("recall", rec),
            ("f1", f1)
        ]
        logger.info("***** Eval results {} *****".format(prefix))

        for eval in coref_evaluator.evaluators:
            prec, rec, f1 = eval.get_prf()
            eval_name = eval.get_name()
            logger.info(f"{eval_name}: precision: {prec:.3}, recall: {rec:.3}, f1: {f1:.3}")

        for key, values in results:
            if isinstance(values, float):
                logger.info(f"  {key} = {values:.3f}")
            else:
                logger.info(f"  {key} = {values}")
            if tb_writer is not None and global_step is not None:
                tb_writer.add_scalar(key, values, global_step)

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

        # TODO: later.
        # if official:
        #     with open(os.path.join(self.args.output_dir, "preds.jsonl"), "w") as f:
        #         f.write(json.dumps(doc_to_prediction) + '\n')
        #         f.write(json.dumps(doc_to_subtoken_map) + '\n')
        #
        #     if self.args.conll_path_for_eval is not None:
        #         conll_results = evaluate_conll(self.args.conll_path_for_eval, doc_to_prediction, doc_to_subtoken_map)
        #         official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
        #         logger.info('Official avg F1: %.4f' % official_f1)

        return results


        # coref_evaluator = CorefEvaluator()
        # for (doc_key, subtoken_maps), batch in val_loader:
        #
        #     batch = tuple(tensor.to(self.args.device) for tensor in batch)
        #     input_ids, attention_mask, gold_clusters = batch
        #
        #     with torch.no_grad():
        #         # outputs = model(input_ids=input_ids,
        #         #                 attention_mask=attention_mask,
        #         #                 gold_clusters=gold_clusters,
        #         #                 return_all_outputs=True)
        #
        #         # TODO: the following code works:
        #         # ids = torch.randint(3, 5, (2,128))
        #         # mask = torch.randint(3, 5, (2,128))
        #         # generated_ids = model.generate(input_ids = ids, attention_mask = mask, max_length=128, num_beams=2, repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
        #         # generated_ids.shape
        #         # torch.Size([2, 128])
        #
        #         #
        #         # But when using the cached pickled artifacts we get [1, 497], and when using the
        #         # non-cached artifacts, i.e. CorefDataset population, we get [1, 606].
        #         # We need to wrap the data to max sequence length, 128, this was probably used
        #         # when we fine-tuned the model and the model refuses to accept other lengths.
        #         # This (wrapping to 128) should be done in CorefDataset class.
        #         #
        #
        #         generated_ids = model.generate(
        #             input_ids = input_ids,
        #             attention_mask = attention_mask,
        #             max_length=self.args.max_total_seq_len,
        #             num_beams=2,
        #             repetition_penalty=2.5,
        #             length_penalty=1.0,
        #             early_stopping=True
        #             )
        #         preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        #         pass
        #         #target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]

class OfflineGenerativeEvaluator(Evaluator):
    def __init__(self, args, tokenizer):
        Evaluator.__init__(self, args, tokenizer)

    def evaluate(self, model, prefix="", tb_writer=None, global_step=None, official=False):
        import pandas as pd
        val_dataset = pd.read_parquet(self.args.pandas_dataframe)

        if self.args.num_examples:
            val_dataset = val_dataset[:self.args.num_examples]

        if self.eval_output_dir and not os.path.exists(self.eval_output_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(self.eval_output_dir)

        val_params = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 0,
        }

        # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Examples number: %d", len(val_dataset))
        model.eval()

        predictions = []
        actuals = []
        coref_evaluator = CorefEvaluator()
        for _, row in val_dataset.iterrows():
            # TODO: try to fine-tune:
            #       length_penalty
            #       repetition_penalty - for longer sequences
            #       num beams
            #

            preds = row["Generated Text"]
            target = row["output"]

            preds_spans = convert_seq_to_spans(preds)
            target_spans = convert_seq_to_spans(target)

            mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(target_spans)
            mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(preds_spans)
            coref_evaluator.update(preds_spans, target_spans,
                                   mention_to_predicted_clusters,
                                   mention_to_gold_clusters)

            predictions.extend(preds)
            actuals.extend(target)

        for eval in coref_evaluator.evaluators:
            prec, rec, f1 = eval.get_prf()
            eval_name = eval.get_name()
            logger.info(f"{eval_name}: precision: {prec:.3}, recall: {rec:.3}, f1: {f1:.3}")

        prec, rec, f1 = coref_evaluator.get_prf()

        results = []
        results += [
            ("precision", prec),
            ("recall", rec),
            ("f1", f1)
        ]
        logger.info("***** Eval average results {} *****".format(prefix))
        for key, values in results:
            if isinstance(values, float):
                logger.info(f"  {key} = {values:.3f}")
            else:
                logger.info(f"  {key} = {values}")
            if tb_writer is not None and global_step is not None:
                tb_writer.add_scalar(key, values, global_step)

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

        # TODO: later.
        # if official:
        #     with open(os.path.join(self.args.output_dir, "preds.jsonl"), "w") as f:
        #         f.write(json.dumps(doc_to_prediction) + '\n')
        #         f.write(json.dumps(doc_to_subtoken_map) + '\n')
        #
        #     if self.args.conll_path_for_eval is not None:
        #         conll_results = evaluate_conll(self.args.conll_path_for_eval, doc_to_prediction, doc_to_subtoken_map)
        #         official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
        #         logger.info('Official avg F1: %.4f' % official_f1)

        return results


        # coref_evaluator = CorefEvaluator()
        # for (doc_key, subtoken_maps), batch in val_loader:
        #
        #     batch = tuple(tensor.to(self.args.device) for tensor in batch)
        #     input_ids, attention_mask, gold_clusters = batch
        #
        #     with torch.no_grad():
        #         # outputs = model(input_ids=input_ids,
        #         #                 attention_mask=attention_mask,
        #         #                 gold_clusters=gold_clusters,
        #         #                 return_all_outputs=True)
        #
        #         # TODO: the following code works:
        #         # ids = torch.randint(3, 5, (2,128))
        #         # mask = torch.randint(3, 5, (2,128))
        #         # generated_ids = model.generate(input_ids = ids, attention_mask = mask, max_length=128, num_beams=2, repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
        #         # generated_ids.shape
        #         # torch.Size([2, 128])
        #
        #         #
        #         # But when using the cached pickled artifacts we get [1, 497], and when using the
        #         # non-cached artifacts, i.e. CorefDataset population, we get [1, 606].
        #         # We need to wrap the data to max sequence length, 128, this was probably used
        #         # when we fine-tuned the model and the model refuses to accept other lengths.
        #         # This (wrapping to 128) should be done in CorefDataset class.
        #         #
        #
        #         generated_ids = model.generate(
        #             input_ids = input_ids,
        #             attention_mask = attention_mask,
        #             max_length=self.args.max_total_seq_len,
        #             num_beams=2,
        #             repetition_penalty=2.5,
        #             length_penalty=1.0,
        #             early_stopping=True
        #             )
        #         preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        #         pass
        #         #target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]

