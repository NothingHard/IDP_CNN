import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from vgg16 import VGG16
from utils import CIFAR10

def main()
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_dir', type=str, default='save', help='directory to store checkpointed models')
	parser.add_argument('--dataset', type=str, default='CIFAR-10', help='dataset in use')
    parser.add_argument('--prof_type', type=str, default='all-one', help='type of profile coefficient')
    parser.add_argument('--output', type=str, default='output.csv', help='output filename (csv)')
    parser.add_argument('--atp', type=int, default=0, help='alternative training procedure')
    # parser.add_argument('--log_dir', type=str, default='log', help='directory containing log text')
    # parser.add_argument('--note', type=str, default='', help='argument for taking notes')

	FLAG = parser.parse_args()
	test(FLAG)

def test(FLAG):
    print("Reading dataset...")
    if FLAG.dataset is 'CIFAR-10':
        dataset = CIFAR10()
    elif: FLAG.dataset is 'CIFAR-100':
        dataset = CIFAR100()
    else:
        raise ValueError("dataset should be either CIFAR-10 or CIFAR-100.")

    Xtest, Ytest = dataset.load_test_data()

    print("Build VGG16 models...")
    dp = [(i+1)*0.05 for i in range(1,20)]
    vgg16 = VGG16("/home/cmchang/IDP_CNN/vgg16.npy", infer=True)
    vgg16.build(dp=dp, prof_type=FLAG.prof_type)

    with tf.Session() as sess:
        if FLAG.save_dir is not None:
            tf.global_variables_initializer.run()
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(FLAG.save_dir)

			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				print("Model restored %s" % ckpt.model_checkpoint_path)
			sess.run(tf.global_variables())
            print("Initialized")

            output = {}
            for dp_i in dp:
                accu = sess.run(vgg16.accu_dict[str(dp_i)], feed_dict={vgg16.x: Xtest, vgg16.y: Ytest})
                output[str(dp_i)] = accu
                print("At DP={dp:.4f}, accu={perf:.4f}".format(dp=dp_i, perf=accu))
            res = pd.DataFrame.from_dict(output)
            res.to_csv(FLAG.output, index=False)
            print("Write into %s" % FLAG.output)