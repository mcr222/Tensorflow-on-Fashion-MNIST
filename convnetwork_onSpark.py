
def main_fun(argv, ctx):
    import tensorflow as tf
#     from hops import tensorboard
#     from hops import hdfs
    from tensorflow.examples.tutorials.mnist import input_data
    
    args = (0.75,0.01,"Adam")
    train_prob_keep, init_learn_rate, optimizer = args

    # load data
    mnist = input_data.read_data_sets('input/fashion', one_hot=True)
    
    tf.set_random_seed(0)
    
    XX = tf.placeholder(tf.float32, shape=[None,784])
    
    #If one component of `shape` is the special value -1, the size of that dimension
    #is computed so that the total size remains constant.
    X = tf.reshape(XX, shape= [-1,28,28,1])
    
    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1))
    B1 = tf.Variable(tf.zeros([4]))
    W2 = tf.Variable(tf.truncated_normal([5, 5, 4, 8], stddev=0.1))
    B2 = tf.Variable(tf.zeros([8]))
    W3 = tf.Variable(tf.truncated_normal([4, 4, 8, 12], stddev=0.1))
    B3 = tf.Variable(tf.zeros([12]))
    W4 = tf.Variable(tf.truncated_normal([588, 200], stddev=0.1))
    B4 = tf.Variable(tf.zeros([200]))
    W5 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
    B5 = tf.Variable(tf.zeros([10]))
    
    
    # 2. Define the model
    #Note that to define a single value placeholder, a scalar, shape is () or [], not 0
    pkeep = tf.placeholder(tf.float32,())
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding="SAME")+B1)
    Y1d = tf.nn.dropout(Y1, pkeep)
    #as strides every 2x2 steps, only half of the values of the previous output 28x28 are kept: 14x14
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1d, W2, strides=[1,2,2,1],padding="SAME")+B2)
    Y2d = tf.nn.dropout(Y2, pkeep)
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2d, W3, strides=[1,2,2,1],padding="SAME")+B3)
    Y3d = tf.nn.dropout(Y3, pkeep)
    #588=7x7x12 (12 output channels)
    Y3d_reshape = tf.reshape(Y3d, shape=[-1,588])
    Y4 = tf.nn.relu(tf.matmul(Y3d_reshape, W4) + B4)
    Y4d = tf.nn.dropout(Y4, pkeep)
    
    Ylogits = tf.matmul(Y4d, W5) + B5
    Y = tf.nn.softmax(Ylogits)
    
    #logits: Unscaled log probabilities.
    # 3. Define the loss function
    Y_ = tf.placeholder(tf.float32)
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = Ylogits, labels= Y_) # calculate cross-entropy with logits
    cross_entropy = tf.reduce_mean(cross_entropy)
    
    # 4. Define the accuracy 
    #predictions is an array of booleans, needs to be casted to a number (True->1, False->0)
    predictions = tf.equal(tf.argmax(Y, 1),tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions,tf.float32))
    
    # 5. Define an optimizer
    
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(init_learn_rate, global_step,100000,0.96)

    if(optimizer == "Adam"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
    else:
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    
    
    # initialize
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    print "Training model with keep prob: " + str(train_prob_keep)+ ", learning rate: " + str(init_learn_rate) + " and optimizer " + optimizer
      
    train_a = []
    train_c = []
    test_a = []
    test_c = []
    
    training_iter = 10000
    #training_iter = 100
    epoch_size = 100
    for i in range(training_iter):
        test = False
        if i % epoch_size == 0:
            print i
            test = True
        
        batch_X, batch_Y = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={XX: batch_X, Y_: batch_Y, pkeep: train_prob_keep})
    
        btrain_a = []
        btrain_c = []
        btest_a = []
        btest_c = []
        if test:
            a, c = sess.run([accuracy, cross_entropy], feed_dict={XX: batch_X, Y_: batch_Y, pkeep: 1})
            btrain_a.append(a)
            btrain_c.append(c)
    
        if test:
            a, c = sess.run([accuracy, cross_entropy], feed_dict={XX: mnist.test.images, Y_: mnist.test.labels, pkeep: 1})
            btest_a.append(a)
            btest_c.append(c)
        
        train_a += btrain_a
        train_c += btrain_c
        test_a += btest_a
        test_c += btest_c
        
    # 7. Plot and visualise the accuracy and loss
    print("Final accuracy: " + str(test_a[-1]))
    print("Final cross entropy: " + str(test_c[-1]))
    #TODO: write to tensorboard logs


from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from com.yahoo.ml.tf import TFCluster, TFNode
from datetime import datetime
import timeit

start_time = timeit.default_timer()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
args, rem = parser.parse_known_args()

sc = SparkContext(conf=SparkConf().setAppName("your_app_name"))
num_executors = int(sc._conf.get("spark.executor.instances"))
num_ps = 17
#TODO add tensorboard
tensorboard = False

cluster = TFCluster.run(sc, main_fun, sys.argv, num_executors, num_ps, tensorboard, TFCluster.InputMode.TENSORFLOW)
cluster.shutdown()
elapsed = timeit.default_timer() - start_time
print "Elapsed time: " + str(elapsed)


