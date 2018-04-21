#!/usr/bin/python
import tensorflow as tf
import time
import os
import math
import json
import matplotlib.pyplot as plt

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data
from utils import vocabulary
from utils.coco.coco import COCO
from utils.coco.pycocoevalcap.eval import COCOEvalCap
from utils.misc import ImageLoader, CaptionData, TopN

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "../data/flickr8k/train-?????-of-00064",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("checkpoint_dir", "../output/model",
                       "Directory containing model checkpoints.")
tf.flags.DEFINE_string("eval_dir", "../output/eval", 
                       "Directory to write event logs.")
tf.flags.DEFINE_string("vocab_file", "../data/flickr8k/word_counts.txt", 
                       "Text file containing the vocabulary.")

tf.flags.DEFINE_integer("eval_interval_secs", 600,
                        "Interval between evaluation runs.")
tf.flags.DEFINE_integer("num_eval_examples", 10132,
                        "Number of examples for evaluation.")

tf.flags.DEFINE_integer("min_global_step", 5000,
                        "Minimum global step to run evaluation.")

tf.flags.DEFINE_integer("beam_size", 3,
                        "The size of beam search for caption generation")

tf.flags.DEFINE_boolean("save_eval_result_as_image", False,
                        "Turn on to save captioned images. if True, please \
                        specified eval_result_dir and val_raw_image_dir")
tf.flags.DEFINE_string("eval_result_dir", None, 
                       "Directory to save captioned images and captions.")
tf.flags.DEFINE_string("val_raw_image_dir", None, 
                       "Directory that stores raw images.")

tf.logging.set_verbosity(tf.logging.INFO)

def evaluate_model(sess, model, vocab, global_step, summary_writer):
  """Computes perplexity-per-word over the evaluation dataset.

  Summaries and perplexity-per-word are written out to the eval directory.

  Args:
    sess: Session object.
    model: Instance of ShowAndTellModel; the model to evaluate.
    global_step: Integer; global step of the model checkpoint.
    summary_writer: Instance of FileWriter.
    summary_op: Op for generating model summaries.
  """

  # Compute perplexity over the entire dataset.
  num_eval_batches = int(
      math.ceil(FLAGS.num_eval_examples / model.config.batch_size))

  start_time = time.time()
  # for perplexity calculation
  sum_losses = 0
  sum_length = 0
  results = {}
  eval_gt = {}
  for i in range(num_eval_batches):
    # current batch sample
    filenames, image_ids, caps,box = sess.run([
                model.filenames, model.image_ids, model.captions, model.bounding_box
                ])
    # print(caps,type(caps),caps.dtype,caps.shape)
    # print(box,type(box),box.dtype)
    # generate batch captions
    caption_data = model.beam_search(sess, vocab)
    
    # generate caption in order to caluculate bleu-1 to blue-4 and cider etc
    for l in range(len(caption_data)):
        word_idxs = caption_data[l][0].sentence
        score = caption_data[l][0].score
        sum_losses += score
        sum_length += len(word_idxs)
        caption = vocab.get_sentence(word_idxs)
        results[image_ids[l]] = [{'caption':caption}]
        eval_gt[image_ids[l]] = [{'caption':byte_str.decode()} for byte_str in caps[l]]

        # Save the result in an image file, if requested
        if FLAGS.save_eval_result_as_image:
            image_file = filenames[l].decode()
            image_name = image_file.split(os.sep)[-1]
            img = plt.imread(os.path.join(FLAGS.val_raw_image_dir,image_name))

            plt.imshow(img)
            plt.axis('off')
            plt.title(caption)
            plt.savefig(os.path.join(FLAGS.eval_result_dir,
                            os.path.splitext(image_name)[0]+'_result.jpg'))

    if not i % 100:
      tf.logging.info("Computed scores for %d of %d batches.", i + 1,
                      num_eval_batches)
  
  # # comment due to json does not support integer key
  # fp = open('%s-%d'%(os.path.join(FLAGS.eval_result_dir,'results.json'),global_step), 'w')
  # json.dump(results, fp)
  # fp.close()

  # Evaluate these captions. Caculate blue-4, metor and cider etc
  # eval_gt = json.load(open(model.config.eval_caption_file))
  # eval_result = json.load(open(model.config.eval_result_file))
  scorer = COCOEvalCap()
  result = scorer.evaluate(eval_gt, results)
  print(result)
  perplexity = sum_losses / sum_length

  # def add_summary(tag, value):
  #   summary = tf.Summary()
  #   value = summary.value.add()
  #   value.simple_value = value
  #   value.tag = tag
  #   summary_writer.add_summary(summary, global_step)

  # # Log perplexity to the FileWriter.
  # add_summary("Perplexity", perplexity)
  # for (k,v) in aDict.items():  
  #   add_summary(k, v)    
  # # Write the Events file to the eval directory.
  # summary_writer.flush()

  eval_time = time.time() - start_time
  tf.logging.info("Finished evaluation at global step %d, Perplexity = %f (%.2g sec).",
                  global_step, perplexity, eval_time)


def run_once(model,vocab, saver, summary_writer):
  """Evaluates the latest model checkpoint.

  Args:
    model: Instance of ShowAndTellModel; the model to evaluate.
    vocab: Dictionary generated duiring data preparing
    saver: Instance of tf.train.Saver for restoring model Variables.
    summary_writer: Instance of FileWriter.
  """
  model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
  if not model_path:
    tf.logging.info("Skipping evaluation. No checkpoint found in: %s",
                    FLAGS.checkpoint_dir)
    return

  with tf.Session() as sess:
    # Load model from checkpoint.
    tf.logging.info("Loading model from checkpoint: %s", model_path)
    saver.restore(sess, model_path)
    global_step = tf.train.global_step(sess, model.global_step)
    tf.logging.info("Successfully loaded %s at global step = %d.",
                    os.path.basename(model_path), global_step)
    if global_step < FLAGS.min_global_step:
      tf.logging.info("Skipping evaluation. Global step = %d < %d", global_step,
                      FLAGS.min_global_step)
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Run evaluation on the latest checkpoint.
    try:
      evaluate_model(
          sess=sess,
          model=model,
          vocab=vocab,
          global_step=global_step,
          summary_writer=summary_writer)
    except Exception as e:
      tf.logging.error("Evaluation failed.")
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def run():
  """Runs evaluation in a loop, and logs summaries to TensorBoard."""
  # Create the evaluation directory if it doesn't exist.
  eval_dir = FLAGS.eval_dir
  if not tf.gfile.IsDirectory(eval_dir):
    tf.logging.info("Creating eval directory: %s", eval_dir)
    tf.gfile.MakeDirs(eval_dir)

  # Create the result image directory if it doesn't exist.
  if FLAGS.save_eval_result_as_image:
    if not tf.gfile.IsDirectory(FLAGS.eval_result_dir):
      tf.logging.info("Creating eval directory: %s", FLAGS.eval_result_dir)
      tf.gfile.MakeDirs(FLAGS.eval_result_dir)

  # build vocabulary file
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  g = tf.Graph()
  with g.as_default():

    config = Config()
    config.input_file_pattern = FLAGS.input_file_pattern
    config.beam_size = FLAGS.beam_size

    # Build the model for evaluation.
    model = CaptionGenerator(config, mode="eval") 
    model.build()

    # Create the Saver to restore model Variables.
    saver = tf.train.Saver()

    # Create the summary writer.
    summary_writer = tf.summary.FileWriter(eval_dir)

    g.finalize()

    # Run a new evaluation run every eval_interval_secs.
    while True:
      start = time.time()
      tf.logging.info("Starting evaluation at " + time.strftime(
          "%Y-%m-%d-%H:%M:%S", time.localtime()))
      run_once(model,vocab, saver, summary_writer)
      time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
      if time_to_next_eval > 0:
        time.sleep(time_to_next_eval)


def main(unused_argv):
  assert FLAGS.input_file_pattern, "--input_file_pattern is required"
  assert FLAGS.checkpoint_dir, "--checkpoint_dir is required"
  assert FLAGS.eval_dir, "--eval_dir is required"
  assert FLAGS.vocab_file, "--vocab_file is required"

  run()


if __name__ == "__main__":
  tf.app.run()

