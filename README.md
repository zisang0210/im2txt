### Introduction
This neural system for image captioning is roughly based on the paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Xu et al. (ICML2015). The input is an image, and the output is a sentence describing the content of the image. It uses a convolutional neural network to extract visual features from the image, and uses a LSTM recurrent neural network to decode these features into a sentence. A soft attention mechanism is incorporated to improve the quality of the caption. This project is implemented using the Tensorflow library, and allows end-to-end training of both CNN and RNN parts.

### Prerequisites
* **Tensorflow** ([instructions](https://www.tensorflow.org/install/))
* **NumPy** ([instructions](https://scipy.org/install.html))
* **OpenCV** ([instructions](https://pypi.python.org/pypi/opencv-python))
* **Natural Language Toolkit (NLTK)** ([instructions](http://www.nltk.org/install.html))
* **Pandas** ([instructions](https://scipy.org/install.html))
* **Matplotlib** ([instructions](https://scipy.org/install.html))
* **tqdm** ([instructions](https://pypi.python.org/pypi/tqdm))

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
cp ../data/faster_rcnn_resnet50_coco_2018_01_28/exported_graphs/frozen_inference_graph.pb  ../data
```
3. skip if have download coco dataset, else run the following command to get coco
```shell
OUTPUT_DIR="/home/zisang/im2txt"
sh ./dataset/download_mscoco.sh.sh ../data/coco
```
4. get feature for each region proposal(100\*2048)
  - for coco run the following command
```shell
DATASET_DIR="/home/zisang/Documents/code/data/mscoco/raw-data"
OUTPUT_DIR="/home/zisang/im2txt/data/coco"
python ./dataset/build_data.py \
  --graph_path="../data/frozen_inference_graph.pb" \
  --dataset "coco" \
  --train_image_dir="${DATASET_DIR}/train2014" \
  --val_image_dir="${DATASET_DIR}/val2014" \
  --train_captions_file="${DATASET_DIR}/annotations/captions_train2014.json" \
  --val_captions_file="${DATASET_DIR}/annotations/captions_val2014.json" \
  --output_dir="${OUTPUT_DIR}" \
  --word_counts_output_file="${OUTPUT_DIR}/word_counts.txt" 
```
  - for flickr8k
```shell
DATASET_DIR="/home/zisang/Documents/code/data/Flicker8k"
OUTPUT_DIR="/home/zisang/im2txt/data/flickr8k"
python ./dataset/build_data.py \
  --graph_path "../data/frozen_inference_graph.pb" \
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
    --beam_size=3 \
    --save_eval_result_as_image \
    --eval_result_dir='../val/results/' \
    --val_raw_image_dir='/home/zisang/Documents/code/data/Flicker8k/Flicker8k_Dataset'
```
The result will be shown in stdout. Furthermore, the generated captions will be saved in the file `output/val/flickr30_results.json`.

* **Inference:**
You can use the trained model to generate captions for any JPEG images! Put such images in the folder `test/images`, and run a command like this:
```shell
python main.py --phase=test \
    --model_file='../output/models/xxxxxx.npy' \
    --beam_size=3
```
The generated captions will be saved in the folder `test/results`.

### Results
A pretrained model with default configuration can be downloaded [here](https://app.box.com/s/xuigzzaqfbpnf76t295h109ey9po5t8p). This model was trained solely on the COCO train2014 data. It achieves the following BLEU scores on the COCO val2014 data (with `beam size=3`):
* **BLEU-1 = 70.3%**
* **BLEU-2 = 53.6%**
* **BLEU-3 = 39.8%**
* **BLEU-4 = 29.5%**

Here are some captions generated by this model:
![examples](examples/examples.jpg)

### References
* [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044). Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio. ICML 2015.
* [The original implementation in Theano](https://github.com/kelvinxu/arctic-captions)
* [An earlier implementation in Tensorflow](https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow)
* [Microsoft COCO dataset](http://mscoco.org/)
