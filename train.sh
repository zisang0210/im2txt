#!/bin/bash

ATTENTION="fc1 fc2 bias bias2 bias_fc1 bias_fc2 rnn"
NUMBER_OF_STEPS=300000

for att in $ATTENTION;
do
  echo python train.py --input_file_pattern="../data/flickr8k/train-?????-of-00016" \
  --number_of_steps=$NUMBER_OF_STEPS \
  --attention="bias" \
  --optimizer="Adam" \
  --faster_rcnn_file="../data/faster_rcnn_resnet50_coco_2018_01_28/model.ckpt"
  --model_file="../output/model/bias_Adam_60000/model.ckpt-60000"
  --train_dir="../output/model/Joint_${att}_Adam_${NUMBER_OF_STEPS}";
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

DATASET_DIR="../flickr8k"
OUTPUT_DIR="../data/flickr8k"
python ./dataset/build_data.py \
  --graph_path "../data/frozen_faster_rcnn.pb" \
  --dataset "flickr8k" \
  --min_word_count 2 \
  --image_dir "${DATASET_DIR}/Flicker8k_Dataset/" \
  --text_path "${DATASET_DIR}/" \
  --output_dir "${OUTPUT_DIR}" \
  --train_shards 16 \
  --val_shards 1 \
  --test_shards 1 \
  --num_threads 16

tar -cvf result.rar bias_Adam_60000/
scp zshwu@202.114.96.180:/home/zshwu/data/val/results/result.rar ./
tar -xvf result.rar