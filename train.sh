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