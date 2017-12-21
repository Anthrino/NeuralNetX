import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

hid1_node_cnt = 500
hid2_node_cnt = 500
hid3_node_cnt = 500

classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])  # 28*28 pixels flattened
y = tf.placeholder('float')


def neural_net_model(data):
	hid1 = {'weights': tf.Variable(tf.random_normal([784, hid1_node_cnt])),
	        'biases': tf.Variable(tf.random_normal(hid1_node_cnt))}

	hid2 = {'weights': tf.Variable(tf.random_normal([hid1_node_cnt, hid2_node_cnt])),
	        'biases': tf.Variable(tf.random_normal(hid2_node_cnt))}

	hid3 = {'weights': tf.Variable(tf.random_normal([hid2_node_cnt, hid3_node_cnt])),
	        'biases': tf.Variable(tf.random_normal(hid3_node_cnt))}

	output = {'weights': tf.Variable(tf.random_normal([hid3_node_cnt, classes])),
	          'biases': tf.Variable(tf.random_normal(classes))}

	layer1 = tf.matmul(data, hid1['weights']) + hid1['biases']
	layer1 = tf.nn.relu(layer1)  # Rectified linear activation fn

	layer2 = tf.add(tf.matmul(layer1, hid2['weights']), hid2['biases'])
	layer2 = tf.nn.relu(layer2)  # Rectified linear activation fn

	layer3 = tf.add(tf.matmul(layer2, hid3['weights']), hid3['biases'])
	layer3 = tf.nn.relu(layer3)  # Rectified linear activation fn

	output_layer = tf.matmul(layer3, output['weights']), output['biases']

	return output


def train_neural_net(x):
	prediction = neural_net_model(x)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

	optimizer = tf.train.AdamOptimizer().minimize(cost)

	no_epochs = 10

	with tf.Session() as sessn:
		sessn.run(tf.initialize_all_variables())

		# Training Stage
		for epoch in range(no_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples / batch_size)):
				x, y = mnist.train.next_batch(batch_size)
				_, c = sessn.run([optimizer, cost], feed_dict={x: x, y: y})
				epoch_loss += c

			print('Finished Epoch', epoch, '> loss : ', epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy : ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_net(x)
