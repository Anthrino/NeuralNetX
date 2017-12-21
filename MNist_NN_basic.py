import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])  # 28*28 pixels

W = tf.Variable(tf.zeros([784, 10]))  # Matrix - No. of features * No. of nodes in layer
b = tf.Variable(tf.zeros([10]))

init = tf.global_variables_initializer()

# NN model
Y = tf.nn.softmax(tf.matmul(X, W) + b)

Y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))  # loss function

correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Training Step

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

with tf.Session() as sessn:
	sessn.run(init)

	# Training Stage
	for epoch in range(10000):
		
		epoch_loss = 0

		batch_X, batch_Y = mnist.train.next_batch(100)
		train_dict = {X: batch_X, Y_: batch_Y}

		sessn.run(train_step, feed_dict=train_dict)
		a, c = sessn.run([accuracy, cross_entropy], feed_dict=train_dict)

		epoch_loss += c

		# print('Finished Epoch', epoch, '> loss : ', epoch_loss)

	test_data = {X: mnist.test.images, Y_: mnist.test.labels}
	a, c = sessn.run([accuracy, cross_entropy], feed_dict=test_data)
	print('Accuracy : ', a)