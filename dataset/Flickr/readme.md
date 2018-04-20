## Flickr8k数据集的字典及tfrecord创建
Flickr8k.token.txt是数据集的标注文件
每个张图片有0-4共5种标注得分，得分越高的数据描述越准确。
因此建议在Flickr8k的训练集中加如不同得分的batch训练权重不同的机制。label得分为n时权重可以为2^n。



#### 1.create_dictionary.py 创建字典

由Flickr8k.token.txt的label创建。

token_path 是Flickr8k.token.txt的绝对路径

vocabulary_size  是词库大小

python3 create_dictionary.py --token_path=token.txt路径　--vocabulary_size = 字典大小

生成dictionary.json和reverse_dictionary.json两个字典

#### 2.create_tfrecord.py 创建数据文件
两个json字典文件放在和   create_tfrecord.py 同一个文件夹  
调用：
                
python3 create_tfrecord.py --image_data_dir=图片所在文件夹　--output_dir=tfrecord输出路径　--label_map_path=token.txt路径

随机选取6000个样本生成：flickr_train.record
2000个生成：flickr_val.record


<pre><code>
  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(witdh),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/score_0': dataset_util.int64_list_feature(sentences[0]),
      'image/score_1': dataset_util.int64_list_feature(sentences[1]),
      'image/score_2': dataset_util.int64_list_feature(sentences[2]),
      'image/score_3': dataset_util.int64_list_feature(sentences[3]),
      'image/score_4': dataset_util.int64_list_feature(sentences[4]),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8'))
  }
</code></pre>


