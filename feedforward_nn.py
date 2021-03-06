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

#QUESTION: what's the value of relu (it is almost the identity function)

# 2. Define the model
inp = raw_input("Use RELU activation (y/n)? If not sigmoid will be used.")
if inp == "y":
    print("Using RELU for inner layers") 
    #obs: tf.nn.crelu concatenates the relu activation
    Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
    #print("Y1 shape: {shape}".format(shape=Y1.shape))
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
    Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
    Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
else:
    print("Using sigmoid for inner layers") 
    Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
    Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
    Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
    Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)

Ylogits = tf.matmul(Y4, W5) + B5
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
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


def training_step(i, update_test_data, update_train_data):

    if i%1000==0:
        print i
    ####### actual learning 
    # reading batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    # the backpropagation training step
    sess.run(train_step, feed_dict={XX: batch_X, Y_: batch_Y})
    
    ####### evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
    train_a = []
    train_c = []
    test_a = []
    test_c = []
    if update_train_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={XX: batch_X, Y_: batch_Y})
        train_a.append(a)
        train_c.append(c)

    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={XX: mnist.test.images, Y_: mnist.test.labels})
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
plt.plot(x_range, train_a[zoom_point:],'r')
plt.plot(x_range, test_a[zoom_point:])
plt.grid(True)
plt.show()

plt.plot(train_c[zoom_point:],'r')
plt.plot(test_c[zoom_point:])
plt.grid(True)
plt.show()
    
'''
What is the maximum accuracy that you can get in each setting for running your
model with 10000 iterations?

Results with RELU+softmax and Adam Optimizer (10.000 iterations):
    Final accuracy: 0.884
    Final cross entropy: 0.370437

Results with RELU+softmax and Gradient Descent (10.000 iterations):
    Final accuracy: 0.8594
    Final cross entropy: 0.437678

Results with sigmoid+softmax and Adam Optimizer (10.000 iterations):
    Final accuracy: 0.8755
    Final cross entropy: 0.35766

Results with sigmoid+softmax and Gradient Descent (10.000 iterations):
    Final accuracy: 0.8668
    Final cross entropy: 0.378372
'''

'''
Is there a big difference between the convergence rate of the sigmoid and the ReLU
? If yes, what is the reason for the difference?

Yes, RELU converges faster. RELU takes care of the Vanishing gradient problem. Sigmoid
gets saturated with large activation values, meaning that the gradient at that neuron
will approach zero and nothing will be learned effectively. As RELU is linear, it does not
saturate.
'''

'''
What is the reason that we use the softmax in our output layer?

We need to normalize the activation outputs (logits) into a probability distribution.
Softwmax does this by normalizing the logits with an exponential function
(difference with large and small values is increased with the exponential function).
'''

'''
By zooming into the second half of the epochs in accuracy and loss plot, do you
see any strange behaviour? What is the reason and how you can overcome them?
(e.g., look at fluctuations or sudden loss increase after a period of decreasing loss).

It seems that in the second half of the epoch the training keeps minimizing loss on training
data but not on testing data. This is a sign of overfitting.

We can also see that for RELU this tendency is clearer, and this might be because sigmoid tends 
to saturate and "freeze" the learned values for neurons.

To avoid overfitting we must either reduce the size of our network or use regularization techniques
 
'''

