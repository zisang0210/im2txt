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
