#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

import sys
import logging

# define logging
filename = os.path.basename(__file__)
logger = logging.getLogger('logger')
fileHandler = logging.FileHandler('./log/%s.log'%filename)
logger.addHandler(fileHandler)
## Change Log Level
logger.setLevel(logging.DEBUG)
logging.basicConfig(filename='%s.log'%filename, level=logging.DEBUG)


def train_step(x_batch, y_batch):
    """
    A single training step
    training할때의 dropout은 0.5를 준다. 단 검증할때에는 1.0으로 세팅한다.
    """
    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, summaries, loss, accuracy = sess.run(
        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    # summary write는 여기에서 한다.
    train_summary_writer.add_summary(summaries, step)

def dev_step(x_batch, y_batch, writer=None):
    """
    Evaluates model on a dev set
    dropout 은 1.0으로 세팅하여 dropout이 적용이 안되도록 한다. 
    evaluation 에서는 정의한 training procedure의 최종결과인 train_op를 넣지 않고
    x와 y값만을 넣어 검증하게 된다.
    """
    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: 1.0
    }
    step, summaries, loss, accuracy = sess.run(
        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    if writer:
        writer.add_summary(summaries, step)

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
#tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
#tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
tf.flags.DEFINE_string("data_file", "../../nsmc_inter_files/ratings_train.txt", "Data source for test.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
#print("\nParameters:")
logger.info("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    #print("{}={}".format(attr.upper(), value))
    logger.info("{}={}".format(attr.upper(), value))
#print("")


# Data Preparation
# ==================================================

# Load data
#print("Loading data...")
logger.info("Loading data...")
#x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
x_text, y = data_helpers.load_data_and_labels2(FLAGS.data_file)

# Build vocabulary
# 문자열 중 가장 많은 단어의 수를 구한다.
max_document_length = max([len(x.split(" ")) for x in x_text])
# 각 단어들을 map 화 한다.
# (max_document_length 미만인 문자열들의 단어는 0이된다)
# 참고로 python2.x 에선는 제대로 동작하지 않는다고 한다.
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
# y의 크기만한 array를 생성 후, 이를 램덤하게 섞는다. 
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
# if len(y) is 28 , dev_sample_index is -2. 
"""
여기서 dev_sample_index는 - 값을 가진다는것에 주의
>>> a = [1,2,3,4,5]
>>> a[:2]
[1, 2]
>>> a[:-2]
[1, 2, 3]
"""
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

# 명시적으로 메모리에서 삭제
del x, y, x_shuffled, y_shuffled

# Vocabulary size is count of word
#print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
logger.info("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
# Train/Dev : 90%/10%
#print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
logger.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))




# Training
# ==================================================

# 명시된 디바이스가 존재하지 않는 경우 실행 디바이스를 TensorFlow가 자동으로 존재하는 디바이스 중 선택하게 하기.
# 연산과 텐서가 어떤 디바이스에 배치되었는지 알아보기 위해, 세션을 만들 때 log_device_placement 옵션을 True로 설정할 수 있습니다
with tf.Graph().as_default():
    session_conf = tf.ConfigProto( allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        # grads_and_vars : List of gradient, variable
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        #out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        inter_files_dir = "../../nsmc_inter_files"
        out_dir = os.path.abspath(os.path.join(inter_files_dir, "runs", timestamp))
        #print("Writing to {}\n".format(out_dir))
        logger.info("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        # Tensorboard에 쓰기 위해 tf.summary 를 사용
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        # 중간계산결과를 저장하는 방법! 
        # 여기서 설정하고, saver.save 로 저장한다
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Generate batches
        # batches 설명 : 
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        # batches 에는 batch_iter 에서 리턴한 값(batch_iter 함수 주석참조)이 들어있다.
        # batch_iter 함수는 yield를 사용하기 때문에 여기서 for 문을 다시한번더 사용해야 하며
        # batch_iter 는 이부분의 for문 one loop 마다 한번씩 실행되어 값을 리턴한다.
        for batch in batches:
            # zip(*batch) : batch_iter에서 zip으로 묶인 값인 batch를 다시 unzip 하는 의미
            # 이유는 x와 y를 분리하기 위해
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            # global_step에는 학습 단계의 횟수가 저장되어 있다.
            current_step = tf.train.global_step(sess, global_step)
            # 100번마다 evaluation(검증)
            if current_step % FLAGS.evaluate_every == 0:
                #print("\nEvaluation:")
                logger.info("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                #print("")
            # 100번마다 file에 중간결과 저장
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                #print("Saved model checkpoint to {}\n".format(path))
                logger.info("Saved model checkpoint to {}\n".format(path))
