#!/bin/bash
date > t5_base_eval_start.txt
python run_coref.py \
  --output_dir=output \
  --cache_dir=cache \
  --model_type=t5-base \
  --model_name_or_path=model_files_checkpoint_epoch3 \
  --tokenizer_name=model_files_checkpoint_epoch3 \
  --config_name=model_files_checkpoint_epoch3/config.json \
  --train_file=coref/train.english.jsonlines \
  --predict_file=coref/test.english.jsonlines \
  --do_eval \
  --num_train_epochs=129 \
  --logging_steps=500 \
  --save_steps=3000 \
  --eval_steps=1000 \
  --max_seq_length=512 \
  --train_file_cache=coref/blahhhh \
  --predict_file_cache=coref/kukuuuu \
  --amp \
  --normalise_loss \
  --max_total_seq_len=5000 \
  --experiment_name=eval_model \
  --warmup_steps=5600 \
  --adam_epsilon=1e-6 \
  --head_learning_rate=3e-4 \
  --learning_rate=1e-5 \
  --adam_beta2=0.98 \
  --weight_decay=0.01 \
  --dropout_prob=0.3 \
  --save_if_best \
  --top_lambda=0.4 \
  --tensorboard_dir=output/tb \
  --conll_path_for_eval=coref/test \
  --overwrite_output_dir \
  --force-gpu \
  --is-generative \
  --is-offline-eval \
  --pandas-dataframe test_inter_paragraph.parq
date > t5_base_eval_end.txt
