from __future__ import absolute_import, division, print_function

import glob
import logging
import os
import pickle
import shutil
import git
import torch

from transformers import AutoConfig, AutoTokenizer, CONFIG_MAPPING, LongformerConfig, RobertaConfig

from modeling import CoreferenceResolutionModel, EndToEndCoreferenceResolutionModel
from data import get_dataset
from cli import parse_args
from training import train, set_seed
from eval import Evaluator
from utils import write_meta_data

logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    allowed_names = ["longformer", "roberta"]
    assert args.model_type in allowed_names
    assert 0 <= args.pos_coeff <= 1

    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)

    if args.predict_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument.")

    if args.output_dir and os.path.exists(args.output_dir) and \
            os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    if args.overwrite_output_dir and os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device


    with open(os.path.join(args.output_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    try:
        write_meta_data(args.output_dir, args)
    except git.exc.InvalidGitRepositoryError:
        logger.info("didn't save metadata - No git repo!")

    logger.info("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, amp training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.amp)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    config.hidden_dropout_prob = args.encoder_dropout_prob

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if args.model_type == "longformer":
        config_class = LongformerConfig
        base_model_prefix = "longformer"
    else:
        config_class = RobertaConfig
        base_model_prefix = "roberta"

    if args.end_to_end:
        EndToEndCoreferenceResolutionModel.config_class = config_class
        EndToEndCoreferenceResolutionModel.base_model_prefix = base_model_prefix
        model = EndToEndCoreferenceResolutionModel.from_pretrained(args.model_name_or_path,
                                                                   config=config,
                                                                   cache_dir=args.cache_dir,
                                                                   antecedent_loss=args.antecedent_loss,
                                                                   max_span_length=args.max_span_length,
                                                                   seperate_mention_loss=args.seperate_mention_loss,
                                                                   prune_mention_for_antecedents=args.prune_mention_for_antecedents,
                                                                   normalize_antecedent_loss=args.normalize_antecedent_loss,
                                                                   only_joint_mention_logits=args.only_joint_mention_logits,
                                                                   no_joint_mention_logits=args.no_joint_mention_logits,
                                                                   pos_coeff=args.pos_coeff,
                                                                   independent_mention_loss=args.independent_mention_loss,
                                                                   normalise_loss=args.normalise_loss,
                                                                   num_neighboring_antecedents=args.num_neighboring_antecedents,
                                                                   independent_start_end_loss=args.independent_start_end_loss,
                                                                   args=args)
    else:
        CoreferenceResolutionModel.config_class = config_class
        CoreferenceResolutionModel.base_model_prefix = base_model_prefix
        model = CoreferenceResolutionModel.from_pretrained(args.model_name_or_path,
                                                           config=config,
                                                           cache_dir=args.cache_dir,
                                                           antecedent_loss=args.antecedent_loss,
                                                           max_span_length=args.max_span_length,
                                                           seperate_mention_loss=args.seperate_mention_loss,
                                                           prune_mention_for_antecedents=args.prune_mention_for_antecedents,
                                                           normalize_antecedent_loss=not args.not_normalize_antecedent_loss,
                                                           only_joint_mention_logits=args.only_joint_mention_logits,
                                                           no_joint_mention_logits=args.no_joint_mention_logits,
                                                           pos_coeff=args.pos_coeff,
                                                           zero_null_logits=args.zero_null_logits,
                                                           args=args)
        # model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    if args.local_rank == 0:
        # End of barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    sampling_prob = args.sampling_prob
    evaluator = Evaluator(args, tokenizer, sampling_prob)
    # Training
    if args.do_train:
        train_dataset = get_dataset(args, tokenizer, evaluate=False)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, evaluator)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer,
    # you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    results = {}

    if args.do_eval and args.local_rank in [-1, 0]:
        result = evaluator.evaluate(model, prefix="final_evaluation")
        results.update(result)
        return results

    return results


if __name__ == "__main__":
    main()
