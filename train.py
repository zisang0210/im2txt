#!/usr/bin/python
import tensorflow as tf

from config import Config
from model import CaptionGenerator

FLAGS = tf.app.flags.FLAGS


tf.flags.DEFINE_string('input_file_pattern', '/home/hillyess/coco_tfrecord/train-?????-of-00256',
                       'Image feature extracted using faster rcnn and corresponding captions')

tf.flags.DEFINE_string('train_dir', '../output/model',
                       'Model checkpoints and summary save here')

tf.flags.DEFINE_string('model_file', None,
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_string('faster_rcnn_file', None,
                       'The file containing a pretrained Faster R-CNN model')

tf.flags.DEFINE_string("optimizer", "SGD",
                        "Adam, RMSProp, Momentum or SGD")

tf.flags.DEFINE_float("initial_learning_rate", "0.001",
                        "")

tf.flags.DEFINE_float("learning_rate_decay_factor", "0.1",
                        "")
tf.flags.DEFINE_integer("num_steps_per_decay", "10000",
                        "")
tf.flags.DEFINE_float("momentum", "0.9",
                        "")
tf.flags.DEFINE_string("attention", "bias",
                        "fc1, fc2, bias, bias2, bias_fc1, bias_fc2, rnn")
tf.flags.DEFINE_integer("number_of_steps", 20000,
                        "Number of training steps.")
# tf.flags.DEFINE_boolean('train_cnn', False,
#                         'Turn on to train both CNN and RNN. \
#                          Otherwise, only RNN is trained')



tf.logging.set_verbosity(tf.logging.INFO)

def main(argv):
    config = Config()
    config.input_file_pattern = FLAGS.input_file_pattern
    config.optimizer = FLAGS.optimizer
    config.initial_learning_rate = FLAGS.initial_learning_rate
    config.learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
    config.num_steps_per_decay = FLAGS.num_steps_per_decay
    config.momentum = FLAGS.momentum
    config.attention_mechanism = FLAGS.attention
    config.save_dir = FLAGS.train_dir
    
    # Create training directory.
    train_dir = config.save_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = CaptionGenerator(config, mode="train")
        model.build()
 
        if FLAGS.faster_rcnn_file is not None:
            model.load_faster_rcnn_feature_extractor(sess, FLAGS.faster_rcnn_file)
        
        if FLAGS.model_file is not None:
            model.load_model_except_faster_rcnn(sess, FLAGS.model_file)

        # Set up the Saver for saving and restoring model checkpoints.
        saver = tf.train.Saver(max_to_keep=config.max_checkpoints_to_keep)

    sess_config = tf.ConfigProto()

    sess_config.gpu_options.allow_growth = True

    # Run training.
    tf.contrib.slim.learning.train(
        model.opt_op,
        train_dir,
        log_every_n_steps=config.log_every_n_steps,
        graph=g,
        global_step=model.global_step,
        number_of_steps=FLAGS.number_of_steps,

        summary_op=model.summary,
        save_summaries_secs=600,
        save_interval_secs=60000,
        init_fn=None,
        saver=saver,
        session_config=sess_config)

    # model = CaptionGenerator(config, mode="train")
    # model.build()
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
    #         img,cap,mask = sess.run([model.image,model.caption, model.mask])
    #         print(img,type(img),img.shape)
    #         print(cap,type(cap),cap.shape)
    #         print(mask,type(mask),mask.shape)

    #     coord.request_stop()
    #     coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
