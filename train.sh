#!/bin/bash

ATTENTION="fc1 fc2 bias bias2 bias_fc1 bias_fc2"

for att in $ATTENTION;
do
  python train.py --input_file_pattern="../data/flickr8k/train-?????-of-00016" \
  --number_of_steps=30 \
  --attention="${att}" \
  --optimizer="Adam" \
  --train_dir="../output/model/${att}_Adam_30";
done
