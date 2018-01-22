import argparse
import numpy as np
import tensorflow as tf

from progress.bar import Bar
from ipywidgets import IntProgress
from IPython.display import display
from vgg16 import VGG16
from utils import CIFAR10

def main()
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_dir', type=str, default='save', help='directory to store checkpointed models')
	parser.add_argument('--dataset', type=str, default='CIFAR-10', help='dataset in use')
    # parser.add_argument('--prof_type', type=str, default='all-one', help='type of profile coefficient')
    # parser.add_argument('--atp', type=int, default=0, help='alternative training procedure')
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

    Xtrain, Ytrain = dataset.load_training_data()
    Xtest, Ytest = dataset.load_test_data()

    print("Build VGG16 models...")
    dp = [(i+1)*0.05 for i in range(1,20)]
    vgg16 = VGG16("/home/cmchang/IDP_CNN/vgg16.npy")

    # build model using 
    vgg16.build(dp=dp, prof_type=FLAG.prof_type)

    saver = tf.train.Saver(tf.global_variables())
    tasks = ['100', '70', '40', '10']

    # initial task
    obj = vgg16.loss_dict[tasks[0]]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # hyper parameters
        learning_rate = 2e-4
        batch_size = 32
        alpha = 0.5
        early_stop_patience = 4
        min_delta = 0.0001

        # recorder
        epoch_counter = 0
        
        # progress bar
        ptrain = IntProgress()
        pval = IntProgress()
        display(ptrain)
        display(pval)
        ptrain.max = int(Xtrain.shape[0]/batch_size)
        pval.max = int(Xtest.shape[0]/batch_size)
        
        # optimizer
        opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        
        # tensorboard writer
        writer = tf.summary.FileWriter(FLAG.log_dir+FLAG.note, sess.graph)

        while(len(tasks)):
        
            # acquire a new task
            cur_task = tasks[0]
            tasks = tasks[1:]
            new_obj = vgg16.loss_dict[cur_task]
            
            # task-wise loss aggregation
            obj = tf.add(tf.multiply(obj, 1.0-alpha), tf.multiply(new_obj, alpha))
            
            # optimizer
            train_op = opt.minimize(obj)
            
            # re-initialize
            initialize_uninitialized(sess)
            
            # reset due to adding a new task
            patience_counter = 0
            current_best_val_loss = 100000 # a large number
            
            # optimize when the aggregated obj
            while(patience_counter < early_stop_patience):
                stime = time.time()

                bar_train = Bar('Training', max=int(Xtrain.shape[0]/batch_size), suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
                bar_val =  Bar('Validation', max=int(Xtest.shape[0]/batch_size), suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
                
                # training an epoch
                for i in range(int(Xtrain.shape[0]/batch_size)):
                    st = i*batch_size
                    ed = (i+1)*batch_size
                    sess.run([train_op], feed_dict={vgg16.x: Xtrain[st:ed,:,:,:],
                                                    vgg16.y: Ytrain[st:ed,:]})
                    ptrain.value +=1
                    ptrain.description = "Training %s/%s" % (i, ptrain.max)
                    bar_train.next()
                
                # validation
                val_loss = 0
                val_accu = 0
                for i in range(int(Xtest.shape[0]/batch_size)):
                    st = i*batch_size
                    ed = (i+1)*batch_size
                    loss, accu, epoch_summary = sess.run([obj, vgg16.accu_dict[cur_task], vgg16.summary_op],
                                        feed_dict={vgg16.x: Xtest[st:ed,:],
                                                vgg16.y: Ytest[st:ed,:]})
                    val_loss += loss
                    val_accu += accu
                    pval.value += 1
                    pval.description = "Testing %s/%s" % (i, pval.max)

                val_loss = val_loss/pval.max
                val_accu = val_accu/pval.max
                
                # early stopping check
                if (current_best_val_loss - val_loss) > min_delta:
                    current_best_val_loss = val_loss
                    patience_counter = 0
                    checkpoint_path = os.path.join(FLAG.save_dir+FLAG.note, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=epoch_counter)
                else:
                    patience_counter += 1
                
                # shuffle Xtrain and Ytrain in the next epoch
                idx = np.random.permutation(Xtrain.shape[0])
                Xtrain, Ytrain = Xtrain[idx,:,:,:], Ytrain[idx,:]
                
                # epoch end
                writer.add_summary(epoch_summary, epoch_counter)
                epoch_counter += 1
                
                ptrain.value = 0
                pval.value = 0
                bar_train.finish()
                bar_val.finish()
                
                print("Epoch %s (%s), %s sec >> obj loss: %.4f, task at %s: %.4f" % (epoch_counter, patience_counter, round(time.time()-stime,2), val_loss, cur_task, val_accu))

        writer.close()