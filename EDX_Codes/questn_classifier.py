'''

	Hierarchical Question Classifier 

	Two layers : Coarse and Fine classifiers

	C0 (initial set of question type classes) -> // Coarse Classifier (Probability/Density of each class) // -> C1 (Top 5 dense / over threshold classes from C0)

	C1 -> C2 (expansion of coarse classes into subclasses) -> // Fine Classifier (Probability/Density of each subclass) // C3 (Final subclass labels)

	Features : 		POS-Tag
					Chunks
					Head_Chunk: combination of first NP+VP after question word	

 '''


import tensorflow as tf
import trec_input_feed as tif
# ** todo: find method to input TREC dataset for training and testing

# todo: define no of hidden layers for both classifier NNs
hid1_node_cnt = 400
hid2_node_cnt = 200
hid3_node_cnt = 100

# todo: find no.of layers at each level
coarse_classes = 6
fine_classes = 50
batch_size = 100

#todo: Define input and output vector sizes (features)
X = tf.placeholder(tf.float32, [None, 784])  # 28*28 pixels flattened
YC_ = tf.placeholder(tf.float32, [None, 10])
YF_ = tf.placeholder(tf.float32, [None, 10])


# ** todo: Modify both NN definitons change dimension of each hidden layer
	
# NN definiton for Top level classifier
def coarse_classf_NN_model(data):

	hid1 = {'weights': tf.Variable(tf.truncated_normal([784, hid1_node_cnt], stddev=0.1)),
	        'biases': tf.Variable(tf.random_normal([hid1_node_cnt]))}

	hid2 = {'weights': tf.Variable(tf.truncated_normal([hid1_node_cnt, hid2_node_cnt], stddev=0.1)),
	        'biases': tf.Variable(tf.random_normal([hid2_node_cnt]))}

	hid3 = {'weights': tf.Variable(tf.truncated_normal([hid2_node_cnt, hid3_node_cnt], stddev=0.1)),
	        'biases': tf.Variable(tf.random_normal([hid3_node_cnt]))}

	output = {'weights': tf.Variable(tf.truncated_normal([hid3_node_cnt, classes], stddev=0.1)),
	          'biases': tf.Variable(tf.random_normal([classes]))}

	# Rectified linear activation fn
	layer1 = tf.nn.relu(tf.add(tf.matmul(data, hid1['weights']), hid1['biases']))

	layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, hid2['weights']), hid2['biases']))

	layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, hid3['weights']), hid3['biases']))

	output_layer = tf.nn.softmax(tf.add(tf.matmul(layer3, output['weights']), output['biases']))

	return output_layer

# NN definiton for fine level classifier
def fine_classf_NN_model(data):
	hid1 = {'weights': tf.Variable(tf.truncated_normal([784, hid1_node_cnt], stddev=0.1)),
	        'biases': tf.Variable(tf.random_normal([hid1_node_cnt]))}

	hid2 = {'weights': tf.Variable(tf.truncated_normal([hid1_node_cnt, hid2_node_cnt], stddev=0.1)),
	        'biases': tf.Variable(tf.random_normal([hid2_node_cnt]))}

	hid3 = {'weights': tf.Variable(tf.truncated_normal([hid2_node_cnt, hid3_node_cnt], stddev=0.1)),
	        'biases': tf.Variable(tf.random_normal([hid3_node_cnt]))}

	output = {'weights': tf.Variable(tf.truncated_normal([hid3_node_cnt, classes], stddev=0.1)),
	          'biases': tf.Variable(tf.random_normal([classes]))}

	# Rectified linear activation fn
	layer1 = tf.nn.relu(tf.add(tf.matmul(data, hid1['weights']), hid1['biases']))

	layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, hid2['weights']), hid2['biases']))

	layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, hid3['weights']), hid3['biases']))

	output_layer = tf.nn.softmax(tf.add(tf.matmul(layer3, output['weights']), output['biases']))

	return output_layer


def train_neural_net(X):
	Y = neural_net_model(X)
	
	cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))  # loss function

	correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

	# Training Step

	optimizer = tf.train.GradientDescentOptimizer(0.003)
	train_step = optimizer.minimize(cross_entropy)

	no_epochs = 10

	with tf.Session() as sessn:
		sessn.run(tf.global_variables_initializer())

		# Training Stage
		for epoch in range(no_epochs):
			epoch_loss = 0
			
			for i in range(int(mnist.train.num_examples / batch_size)):

				batch_X, batch_Y = mnist.train.next_batch(batch_size)
				train_dict = {X: batch_X, Y_: batch_Y}

				sessn.run(train_step, feed_dict=train_dict)
				a, c = sessn.run([accuracy, cross_entropy], feed_dict=train_dict)
				# print('Batch',i,' / Cost :', c, ' / Accuracy :', a)
				epoch_loss += c

			print('Finished Epoch', epoch, '> loss : ', epoch_loss)

		test_data = {X: mnist.test.images, Y_: mnist.test.labels}
		a, c = sessn.run([accuracy, cross_entropy], feed_dict=test_data)
		print('Test Accuracy : ', a)


if __name__ == '__main__':
	train_neural_net(X)
