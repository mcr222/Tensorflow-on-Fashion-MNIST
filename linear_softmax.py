# all tensorflow api is accessible through this
import tensorflow as tf        
# to visualize the resutls
import matplotlib.pyplot as plt 
# 70k mnist dataset that comes with the tensorflow container
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(0)

# load data
mnist = input_data.read_data_sets('input/fashion', one_hot=True)

# 1. Define Variables and Placeholders
# None means that a dimension can be of any length
# the first dimension is the # of input of flatten images
XX = tf.placeholder(tf.float32, shape=[None,784])

#W contains the 784 weights (in each column) for each of the
#10 linear regressions
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 2. Define the model
Y = tf.matmul(XX, W)+b
model = tf.nn.softmax(Y)

# 3. Define the loss function 
#note that Y_ will have dimension [None,10] 
Y_ = tf.placeholder(tf.float32)
# default * is element-wise (to multiply matrices use tf.matmul)
#the first dimension of Y_ * tf.log(model) are all the images (None)
#the second dimension is 10, thus the results of the model (that 
#have to be summed for each image (index [1] is the 2nd dimension)

#Cross entropy formula is for one example, as we train in batches, the
#loss is the mean of all individual cross entropies.

#NOTE: using this manual approach to compute cross entropy with AdamOptimizer results in 
#unstable results and accuracy drops to almost 0 and cross entropy to NaN
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(model), reduction_indices=[1]))

#This is a more numerically stable way of calculating cross entropy by internally computing also
#the softmax layer values (that's why we use Y instead of the model)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))

# 4. Define the accuracy 
#predictions is an array of booleans, needs to be casted to a number (True->1, False->0)
predictions = tf.equal(tf.argmax(model, 1),tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(predictions,tf.float32))

# 5. Define an optimizer
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

'''
Results with Gradient Descent (10.000 iterations):
    Final accuracy: 0.85
    Final cross entropy: 0.426961

Results with Adam Optimizer (10.000 iterations):
    Final accuracy: 0.85
    Final cross entropy: 0.484273

'''

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
print("Final accuracy: " + str(train_a[-1]))
print("Final cross entropy: " + str(train_c[-1]))

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
    
