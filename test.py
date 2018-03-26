#! /usr/bin/env python

import tensorflow as tf
import data_helpers

#tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
#tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/test.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("data_file", "../nsmc/small.txt", "Data source for test.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

#x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
x_text, y = data_helpers.load_data_and_labels2(FLAGS.data_file)
#x_text, y = data_helpers.load_data_and_labels2(FLAGS.positive_data_file)

print (x_text)
