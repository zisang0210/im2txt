#!/usr/bin/python
import tensorflow as tf

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data

FLAGS = tf.app.flags.FLAGS


# tf.flags.DEFINE_string('model_file', None,
#                        'If sepcified, load a pretrained model from this file')


tf.flags.DEFINE_string('optimizer', 'Adam',
                       'The file containing a pretrained CNN model')

# tf.flags.DEFINE_boolean('train_cnn', False,
#                         'Turn on to train both CNN and RNN. \
#                          Otherwise, only RNN is trained')



tf.logging.set_verbosity(tf.logging.INFO)

def main(argv):
    config = Config()
    config.phase = 'train'
    config.optimizer = FLAGS.optimizer

    model = CaptionGenerator(config)
    model.train()

    # with tf.Session() as sess:
    # #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.local_variables_initializer())
    # #     if model.init_fn:
    # #         model.init_fn(sess)
        
    

    #     # Start populating the filename queue.
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)

    #     for i in range(1):
    #         # Retrieve a single instance:
    #         example = sess.run(model.images)
    #         print(example,type(example),example.shape)

    #     coord.request_stop()
    #     coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
