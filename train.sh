
for att in "fc1 fc2 bias bias2 bias_fc1 bias_fc2 rnn";
do
python train.py --input_file_pattern='../data/flickr8k/train-?????-of-00016' \
    --number_of_steps=30 \    
    --attention='${att}' \    
    --optimizer='Adam' \    
    --train_dir='../output/model/${att}_Adam_10000';
done
