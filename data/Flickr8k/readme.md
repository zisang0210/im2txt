## Flickr8k数据集的字典及tfrecord创建
Flickr8k.token.txt是数据集的标注文件
每个张图片有0-4共5种标注得分，得分越高的数据描述越准确。
因此建议在Flickr8k的训练集中加如不同得分的batch训练权重不同的机制。label得分为n时权重可以为2^n。



#### 1.create_dictionary.py 创建字典

由Flickr8k.token.txt的label创建。

token_path 是Flickr8k.token.txt的绝对路径

vocabulary_size = 10000 是词库大小

生成dictionary.json和reverse_dictionary.json两个字典

#### 2.create_tfrecord.py 创建数据文件
修改这路径三个参数
data_dir
output_dir
label_map_path
随机选取6000个样本生成：
flickr_train.record
2000个生成：
flickr_val.record


<pre><code>
  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(witdh),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/score': dataset_util.bytes_feature(importance),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/class/sentence': dataset_util.int64_list_feature(sentence)
  }
</code></pre>


