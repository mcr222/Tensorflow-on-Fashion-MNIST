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

#If one component of `shape` is the special value -1, the size of that dimension
#is computed so that the total size remains constant.
X = tf.reshape(XX, shape= [-1,28,28,1])

#Alternative way of providing a decaying learning rate is to provide a scalar placeholder
#and then on running training provide a value via feed_dict (value can be updated as we want).
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.02, global_step,100000,0.96)


'''
W1 will be convoluted with the input image. Note that there will be 4 separate convolutions, since the 
output dimension of W1 is 4, which means that there are 4 convolution matrices.
For discrete, two-dimensional variables A and B, the following equation defines the convolution of A and B:

C(j,k)=sum_p(sum_q(A(p,q)B(j-p+1,k-q+1)))

p and q run over all values that lead to legal subscripts of A(p,q) and B(j-p+1,k-q+1).
This leads to the convolution C = conv2(A,B) has size size(A)+size(B)-1. 

I think in a convolutional layer the convolution is only applied to (j,k) indexes of the input
thus obtaining a convoluted matrix of the same size as the input. 
'''
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1))
B1 = tf.Variable(tf.zeros([4]))
'''
TODO: CLARIFY W2, for each input channel (4) has 8 matrices, this would
generate 4*8=32 output channels (but code seems to output 8 channels).
'''
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
#TODO: check padding
#The output of this first convolution are 28x28 matrices with 4 channels each.
#Then B1 is added to each of the channels.
#TODO: how does addition work?? It seems that it would add the same value to 
#each of the values of one channel matrix (4 channels and 4 bias values, 1 bias added to the whole channel matrix)
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
#learning_rate = tf.train.exponential_decay(0.5)
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Passing global_step to minimize() will increment it at each step.
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)


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
    
