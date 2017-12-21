
# Convolutional Neural Network for MNIST dataset classification

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

opch_L1 = 4
opch_L2 = 8
opch_L3 = 12
opch_L4 = 200
classes = 10
batch_size = 100

X = tf.placeholder(tf.float32, [None, 28, 28, 1])  # 28*28 pixels grayscale
Y_ = tf.placeholder(tf.float32, [None, 10])


def neural_net_model(data):

	L1 = {'weights': tf.Variable(tf.truncated_normal([5, 5, 1, opch_L1], stddev=0.1)),
	        'biases': tf.Variable(tf.ones([opch_L1])/10)}

	L2 = {'weights': tf.Variable(tf.truncated_normal([5, 5, opch_L1, opch_L2], stddev=0.1)),
	        'biases': tf.Variable(tf.ones([opch_L2])/10)}

	L3 = {'weights': tf.Variable(tf.truncated_normal([4, 4, opch_L2, opch_L3], stddev=0.1)),
	        'biases': tf.Variable(tf.ones([opch_L3])/10)}

	L4 = {'weights': tf.Variable(tf.truncated_normal([7*7*opch_L3, opch_L4], stddev=0.1)),
	          'biases': tf.Variable(tf.ones([opch_L4])/10)}

	output = {'weights': tf.Variable(tf.truncated_normal([opch_L4, classes], stddev=0.1)),
	          'biases': tf.Variable(tf.zeros([classes])/10)}


	# Rectified linear activation fn and softmax
	layer1 = tf.nn.relu(tf.add(tf.nn.conv2d(data, L1['weights'], strides=[1, 1, 1, 1], padding='SAME'), L1['biases']))

	layer2 = tf.nn.relu(tf.add(tf.nn.conv2d(layer1, L2['weights'], strides=[1, 2, 2, 1], padding='SAME'), L2['biases']))

	layer3 = tf.nn.relu(tf.add(tf.nn.conv2d(layer2, L3['weights'], strides=[1, 2, 2, 1], padding='SAME'), L3['biases']))

	layer4 = tf.nn.relu(tf.add(tf.matmul(tf.reshape(layer3, shape=[-1, 7*7*opch_L3]), L4['weights']), L4['biases']))

	output_layer = tf.nn.softmax(tf.add(tf.matmul(layer4, output['weights']), output['biases']))

	return output_layer


def train_neural_net(X):
	Y = neural_net_model(X)
	
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=Y_)  # loss function
	cross_entropy = tf.reduce_mean(cross_entropy)*100

	correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

	# Training Step

	optimizer = tf.train.GradientDescentOptimizer(0.003)
	train_step = optimizer.minimize(cross_entropy)

	no_epochs = 1

	with tf.Session() as sessn:
		sessn.run(tf.global_variables_initializer())

		# Training Stage
		for epoch in range(no_epochs):
			epoch_loss = 0
			
			for i in range(int(mnist.train.num_examples / batch_size)):

				batch_X, batch_Y = mnist.train.next_batch(batch_size)
				batch_X = np.reshape(batch_X, (-1, 28, 28, 1))
				train_dict = {X: batch_X, Y_: batch_Y}

				sessn.run(train_step, feed_dict=train_dict)
				a, c = sessn.run([accuracy, cross_entropy], feed_dict=train_dict)
				# print('Batch',i,' / Cost :', c, ' / Accuracy :', a)
				epoch_loss += c

			print('Finished Epoch', epoch, '> loss : ', epoch_loss)

		test_data = {X: np.reshape(mnist.test.images, (-1, 28, 28, 1)), Y_: mnist.test.labels}
		a, c = sessn.run([accuracy, cross_entropy], feed_dict=test_data)
		print('Test Accuracy : ', a)


if __name__ == '__main__':
	train_neural_net(X)
