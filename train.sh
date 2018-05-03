#!/bin/bash

ATTENTION="fc1 fc2 bias bias2 bias_fc1 bias_fc2 rnn"
NUMBER_OF_STEPS=30

for att in $ATTENTION;
do
  echo python train.py --input_file_pattern="../data/flickr8k/train-?????-of-00016" \
  --number_of_steps=$NUMBER_OF_STEPS \
  --attention="${att}" \
  --optimizer="Adam" \
  --train_dir="../output/model/${att}_Adam_${NUMBER_OF_STEPS}";
done


python train.py --input_file_pattern="../data/flickr8k/train-?????-of-00016" \
  --number_of_steps=120000 \
  --attention="bias" \
  --optimizer="Momentum" \
  --momentum=0.9 \
  --initial_learning_rate=0.001 \
  --learning_rate_decay_factor=0.1 \
  --num_steps_per_decay=40000 \
  --train_dir="../output/model/bias_Momentum_lr_0.0005_decay_0.2_40000"

python eval.py --input_file_pattern='../data/flickr8k/val-?????-of-00008' \
    --checkpoint_dir='../output/model/bias_Adam_60000' \
    --attention='bias' \
    --eval_dir='../output/eval/bias_Adam_60000' \
    --min_global_step=10 \
    --num_eval_examples=32 \
    --vocab_file="../data/flickr8k/word_counts.txt" \
    --beam_size=3 \
    --save_eval_result_as_image \
    --eval_result_dir='../val/results/bias_Adam_60000' \
    --val_raw_image_dir='../flickr8k/Flicker8k_Dataset'