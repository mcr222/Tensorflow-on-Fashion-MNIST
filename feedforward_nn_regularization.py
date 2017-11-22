# all tensorflow api is accessible through this
import tensorflow as tf        
# to visualize the resutls
import matplotlib.pyplot as plt 
# 70k mnist dataset that comes with the tensorflow container
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(0)

# load data
mnist = input_data.read_data_sets('input/fashion', one_hot=True)

XX = tf.placeholder(tf.float32, shape=[None,784])

'''
QUESTION: if we have dropout (specially if high), can we reuse training examples to train? With dropout we essentially block 
    some neurons, so we are kind of limiting the input features of one example. Therefore, if we retrain the example on
    another network (differently blocked neurons) it might add extra information since other features will be blocked.
    OBS: similar to the approach of blocking parts of an image when training.
'''

'''
It requires a `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.
  
It needs global_step as it is every time independently called (no memory i guess)
'''
#Alternative way of providing a decaying learning rate is to provide a scalar placeholder
#and then on running training provide a value via feed_dict (value can be updated as we want).
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.02,global_step,100000,0.96)

# truncated_normal generates a truncated normal distribution as initializer of variables
#Weights initialized with small random values between -0.2 and +0.2
W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1)) # 784 = 28 * 28
B1 = tf.Variable(tf.zeros([200]))
W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
B2 = tf.Variable(tf.zeros([100]))
W3 = tf.Variable(tf.truncated_normal([100, 60], stddev=0.1))
B3 = tf.Variable(tf.zeros([60]))
W4 = tf.Variable(tf.truncated_normal([60, 30], stddev=0.1))
B4 = tf.Variable(tf.zeros([30]))
W5 = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))


# 2. Define the model
#Note that to define a single value placeholder, a scalar, shape is () or [], not 0
pkeep = tf.placeholder(tf.float32,())

inp = raw_input("Use RELU activation (y/n)? If not sigmoid will be used.")
if inp == "y":
    print("Using RELU for inner layers") 
    #obs: tf.nn.crelu concatenates the relu activation
    Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
    '''
    With probability `keep_prob`, outputs the input 
    element scaled up by `1 / keep_prob`, otherwise outputs `0`.  The 
    scaling is so that the expected sum is unchanged.
    '''
    Y1d = tf.nn.dropout(Y1, pkeep)
    #print("Y1 shape: {shape}".format(shape=Y1.shape))
    Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
    Y2d = tf.nn.dropout(Y2, pkeep)
    Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
    Y3d = tf.nn.dropout(Y3, pkeep)
    Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
    Y4d = tf.nn.dropout(Y4, pkeep)
else:
    print("Using sigmoid for inner layers") 
    Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
    Y1d = tf.nn.dropout(Y1, pkeep)
    Y2 = tf.nn.sigmoid(tf.matmul(Y1d, W2) + B2)
    Y2d = tf.nn.dropout(Y2, pkeep)
    Y3 = tf.nn.sigmoid(tf.matmul(Y2d, W3) + B3)
    Y3d = tf.nn.dropout(Y3, pkeep)
    Y4 = tf.nn.sigmoid(tf.matmul(Y3d, W4) + B4)
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
#learning_rate = tf.train.exponential_decay(0.5)
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Passing global_step to minimize() will increment it at each step.
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
'''
Global step can also be updated manually using:
    global_step = tf.assign(global_step, global_step+1)
    sess.run(global_step)

OBS: NOT sure if sess.run(global_step) is necessary to update the value (or if when it is called it
already updates). 
'''


# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

'''
Results with RELU+softmax and Adam Optimizer (10.000 iterations):
    Final accuracy: 0.884
    Final cross entropy: 0.370437

Results with RELU+softmax and Gradient Descent (10.000 iterations):


Results with sigmoid+softmax and Adam Optimizer (10.000 iterations):


Results with sigmoid+softmax and Gradient Descent (10.000 iterations):
   
'''


def training_step(i, update_test_data, update_train_data):

    if i%1000==0:
        print i
    ####### actual learning 
    # reading batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    # the backpropagation training step
    sess.run(train_step, feed_dict={XX: batch_X, Y_: batch_Y, pkeep: 0.75})
    
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


# 6. Train and test the model, store the accuracy and loss per iteration

train_a = []
train_c = []
test_a = []
test_c = []
    
training_iter = 10000
epoch_size = 100
for i in range(training_iter):
    test = False
    if i % epoch_size == 0:
        test = True
    a, c, ta, tc = training_step(i, test, test)
    train_a += a
    train_c += c
    test_a += ta
    test_c += tc
    
# 7. Plot and visualise the accuracy and loss
print("Final accuracy: " + str(test_a[-1]))
print("Final cross entropy: " + str(test_c[-1]))

# accuracy training vs testing dataset
plt.plot(train_a,'r')
plt.plot(test_a)
plt.grid(True)
plt.show()

# loss training vs testing dataset
plt.plot(train_c,'r')
plt.plot(test_c)
plt.grid(True)
plt.show()

# Zoom in on the tail of the plots
zoom_point = 50
x_range = range(zoom_point,training_iter/epoch_size)
plt.plot(x_range, train_a[zoom_point:])
plt.plot(x_range, test_a[zoom_point:])
plt.grid(True)
plt.show()

plt.plot(train_c[zoom_point:])
plt.plot(test_c[zoom_point:])
plt.grid(True)
plt.show()
    
