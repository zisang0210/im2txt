### Introduction
This neural system for image captioning is motivated by the paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Xu et al. (ICML2015). The input is an image, and the output is a sentence describing the content of the image. It uses faster rcnn model to extract visual features from the image, and uses a LSTM recurrent neural network to decode these features into a sentence. A soft attention mechanism is incorporated to improve the quality of the caption. This project is implemented using the Tensorflow library, and currently allows training of RNN part only.

### Prerequisites
* **Tensorflow** ([instructions](https://www.tensorflow.org/install/))
* **NumPy** ([instructions](https://scipy.org/install.html))
* **OpenCV** ([instructions](https://pypi.python.org/pypi/opencv-python))
* **Natural Language Toolkit (NLTK)** ([instructions](http://www.nltk.org/install.html))
* **Pandas** ([instructions](https://scipy.org/install.html))
* **Matplotlib** ([instructions](https://scipy.org/install.html))

### Usage

* **tips:**
1. delete all pycache folders under current directory
```shell
find . -name '__pycache__' -type d -exec rm -rf {} \;
```

* **Dataset Preparing:**
1. download faster_rcnn_resnet checkpoint
```shell
cd data
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
tar -xzf faster_rcnn_resnet50_coco_2018_01_28.tar.gz
```
2. frozen graph using checkpoint
```shell
cd ../code/
export PYTHONPATH=$PYTHONPATH:./object_detection/
python ./object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ../data/faster_rcnn_resnet50_coco_2018_01_28/pipeline.config --trained_checkpoint_prefix ../data/faster_rcnn_resnet50_coco_2018_01_28/model.ckpt  --output_directory ../data/faster_rcnn_resnet50_coco_2018_01_28/exported_graphs
cp ../data/faster_rcnn_resnet50_coco_2018_01_28/exported_graphs/frozen_inference_graph.pb  ../data/frozen_faster_rcnn.pb
```
3. skip if have download coco dataset, else run the following command to get coco
```shell
OUTPUT_DIR="/home/zisang/im2txt"
sh ./dataset/download_mscoco.sh.sh ../data/coco
```
4. get feature for each region proposal(100\*2048)

for coco run the following command
```shell
DATASET_DIR="/home/zisang/Documents/code/data/mscoco/raw-data"
OUTPUT_DIR="/home/zisang/im2txt/data/coco"
python ./dataset/build_data.py \
  --graph_path="../data/frozen_faster_rcnn.pb" \
  --dataset "coco" \
  --train_image_dir="${DATASET_DIR}/train2014" \
  --val_image_dir="${DATASET_DIR}/val2014" \
  --train_captions_file="${DATASET_DIR}/annotations/captions_train2014.json" \
  --val_captions_file="${DATASET_DIR}/annotations/captions_val2014.json" \
  --output_dir="${OUTPUT_DIR}" \
  --word_counts_output_file="${OUTPUT_DIR}/word_counts.txt" 
```

for flickr8k
```shell
DATASET_DIR="/home/zisang/Documents/code/data/Flicker8k"
OUTPUT_DIR="/home/zisang/im2txt/data/flickr8k"
python ./dataset/build_data.py \
  --graph_path "../data/frozen_faster_rcnn.pb" \
  --dataset "flickr8k" \
  --min_word_count 2 \
  --image_dir "${DATASET_DIR}/Flicker8k_Dataset/" \
  --text_path "${DATASET_DIR}/" \
  --output_dir "${OUTPUT_DIR}" \
  --train_shards 32\
  --num_threads 8
```


* **Training:**
First make sure you are under the folder `code`, then setup various parameters in the file `config.py` and then run a command like this:
```shell
python train.py --input_file_pattern='../data/flickr8k/train-?????-of-00016' \
    --number_of_steps=10000 \
    --attention='bias' \
    --optimizer='Adam' \
    --train_dir='../output/model'
```
To monitor the progress of training, run the following command:
```shell
tensorboard --logdir='../output/model'
```

* **Evaluation:**
To evaluate a trained model using the flickr30 data, run a command like this:
```shell
python eval.py --input_file_pattern='../data/flickr8k/val-?????-of-00008' \
    --checkpoint_dir='../output/model' \
    --attention='bias' \
    --eval_dir='../output/eval' \
    --min_global_step=10 \
    --num_eval_examples=32 \
    --vocab_file="../data/flickr8k/word_counts.txt" \
    --beam_size=3 \
    --save_eval_result_as_image \
    --eval_result_dir='../val/results/' \
    --val_raw_image_dir='/home/zisang/Documents/code/data/Flicker8k/Flicker8k_Dataset'
```
The result will be shown in stdout and stored in eval_dir as tensorflow summary.

* **Inference:**
A web interface was built using [Flask](http://flask.pocoo.org/). You can use the trained model to generate captions for any JPEG images!

1 - Install Flask

```
pip install Flask
```
2 - First get the frozen graph:
```shell
python export.py --model_folder='../output/model' \
    --output_path='../data/frozen_lstm.pb' \
    --attention='bias'
```
Run Flaskr
```
python server.py --mode ours \
    --vocab_path ../data/flickr8k/word_counts.txt
```
or run the following to see our results
```
python server.py --mode att-nic \
    --faster_rcnn_model_file='../data/frozen_faster_rcnn.pb' \
    --lstm_model_file='../data/frozen_lstm.pb' 
    --vocab_file="../data/flickr8k/word_counts.txt" \
```
3 - Picture test interface http://127.0.0.1:5000

4 - Admin log in http://127.0.0.1:5000/admin to see more information

Username: admin
Password: 0000


### Results
This model was trained solely on the COCO train2014 data. It achieves the following BLEU scores on the COCO val2014 data (with `beam size=3`):
* **BLEU-1 = 15.8%**
* **BLEU-2 = 4.9%**
* **BLEU-3 = 1.0%**
* **BLEU-4 = 0%**
* **METEOR = 4.4%**
* **Rouge = 10.1%**
* **CIDEr = 2.5%**
* **Perplexity = 6.4**
compared to Show, Attend and Tell, which have achieved the following performance:
* **BLEU-1 = 70.3%**
* **BLEU-2 = 53.6%**
* **BLEU-3 = 39.8%**
* **BLEU-4 = 29.5%**
there is still a long way to go.
### References
* [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044). Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio. ICML 2015.
* [The original implementation in Theano](https://github.com/kelvinxu/arctic-captions)
* [An earlier implementation in Tensorflow](https://github.com/DeepRNN/image_captioning)
* [Tensorflow models im2txt](https://github.com/tensorflow/models/tree/master/research/im2txt)
