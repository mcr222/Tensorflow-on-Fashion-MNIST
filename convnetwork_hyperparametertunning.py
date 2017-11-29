# all tensorflow api is accessible through this
import tensorflow as tf        
# to visualize the resutls
import matplotlib.pyplot as plt 
# 70k mnist dataset that comes with the tensorflow container
from tensorflow.examples.tutorials.mnist import input_data
from hyperopt import fmin, tpe, hp, rand
import hyperopt

train_step = None
accuracy = None
cross_entropy = None
mnist = input_data.read_data_sets('input/fashion', one_hot=True)
sess = None
XX = None
Y_ = None
pkeep = None

def initialize_convnetwork(init_learn_rate, optimizer):
    global train_step, accuracy, cross_entropy, sess, XX, Y_, pkeep
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


def training_step(i, update_test_data, update_train_data, train_prob_keep):

    if i%1000==0:
        print i
    ####### actual learning 
    # reading batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    # the backpropagation training step
    sess.run(train_step, feed_dict={XX: batch_X, Y_: batch_Y, pkeep: train_prob_keep})
    
    ####### evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
    train_a = []
    train_c = []
    test_a = []
    test_c = []
    if update_train_data:
        #TODO: pkeep: 0.75 or 1? It is not really training, just training evaluation, so I would use 1 because
        #evaluation is always considering the full network, but then accuracy and cross_entropy are not the 
        #real values on the training (since training uses 0.75). On the other hand if we use 0.75 then the accuracy
        #is not the accuracy of the real model (as it is reduced with the dropout) on the training data. 
        a, c = sess.run([accuracy, cross_entropy], feed_dict={XX: batch_X, Y_: batch_Y, pkeep: 1})
        train_a.append(a)
        train_c.append(c)

    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={XX: mnist.test.images, Y_: mnist.test.labels, pkeep: 1})
        test_a.append(a)
        test_c.append(c)

    
    return (train_a, train_c, test_a, test_c)

#Need to use a input tuple so that it can be called by hyperopt
def train_model(args = (0.75,0.01,"Adam")):
    train_prob_keep, init_learn_rate, optimizer = args
    print "Training model with keep prob: " + str(train_prob_keep)+ ", learning rate: " + str(init_learn_rate) + " and optimizer " + optimizer
    initialize_convnetwork(init_learn_rate, optimizer)
      
    train_a = []
    train_c = []
    test_a = []
    test_c = []
    
    #training_iter = 10000
    training_iter = 100
    epoch_size = 100
    for i in range(training_iter):
        test = False
        if i % epoch_size == 0:
            test = True
        a, c, ta, tc = training_step(i, test, test, train_prob_keep)
        train_a += a
        train_c += c
        test_a += ta
        test_c += tc
        
    # 7. Plot and visualise the accuracy and loss
    print("Final accuracy: " + str(test_a[-1]))
    print("Final cross entropy: " + str(test_c[-1]))
    
    return test_a[-1]

#Note that, in order for this to work we need to use networkx v1.11 (with networkx v2.0 does not work)
#see: https://github.com/hyperopt/hyperopt/issues/326
space = hp.choice('args',
      [
          (hp.uniform('train_prob_keep_Adam',0.5,0.8), hp.uniform('learn_rate_Adam',0.001,0.01), 'Adam'),
          (hp.uniform('train_prob_keep_Grad',0.5,0.8), hp.uniform('learn_rate_Grad',0.1,0.8), 'Gradient')
        ])
'''
The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. 
SMBO methods sequentially construct models to approximate the performance of hyperparameters based on 
historical measurements, and then subsequently choose new hyperparameters to test based on this model.
'''
best = fmin(train_model,
    space=space,
    algo= tpe.suggest,
    max_evals=15)
print best
print hyperopt.space_eval(space, best)

#train_model()  
