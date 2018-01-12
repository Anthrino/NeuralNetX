'''
Attention-Based Bi CNN for Answer Sentence Selection from context.


'''

import tensorflow as tf
import numpy as np
import csv
from Helpers import utils

# IMPORT DATASET

# Hyperparameters

opch_L1 = 4
opch_L2 = 8
opch_L3 = 12
opch_L4 = 200
classes = 10
batch_size = 100

sequence_length =
no_classes
vocab_size,
embedding_size, filter_sizes, num_filters

Q = tf.placeholder(tf.float32, [None, 28, 28, 1])
A = tf.placeholder(tf.float32, [None, 28, 28, 1])

Y_ = tf.placeholder(tf.float32, [None, 10])


def _process_input(self, file):

	questions = []
	answers = []
	file = ".\Dataset\WikiQACorpus\WikiQA-dev.tsv"
		
	with open(file, encoding="utf8") as data_file:
		source = list(csv.reader(data_file, delimiter="\t", quotechar='"'))
		q_index = 'Q-1'
		ans_sents = []

		for row in source[1:]:

			if q_index != row[0]:
				answers.append(ans_sents)
				ans_sents = []
				questions.append(row[1])
				q_index = row[0]

		ans_sents.append({row[5]: row[6]})

	answers.append(ans_sents)
	answers = answers[1:]

	for i in range(len(questions)):
		print("Question:", questions[i])
		print("Answers:", answers[i])

	inp_vector = [utils.process_word(word=w,
	                                 word2vec=self.word2vec,
	                                 vocab=self.vocab,
	                                 ivocab=self.ivocab,
	                                 word_vector_size=self.word_vector_size,
	                                 to_return="word2vec") for w in inp]

		# print(inp)
		# print([np.array(wvec).shape for wvec in inp_vector])
		# print(np.array(inp_vector).shape)

		q_vector = [utils.process_word(word=w,
		                               word2vec=self.word2vec,
		                               vocab=self.vocab,
		                               ivocab=self.ivocab,
		                               word_vector_size=self.word_vector_size,
		                               to_return="word2vec") for w in q]
		inputs.append(np.vstack(inp_vector).astype(floatX))
		questions.append(np.vstack(q_vector).astype(floatX))
		if self.mode != 'deploy':
			answers.append(utils.process_word(word=x["A"],
			                                  word2vec=self.word2vec,
			                                  vocab=self.vocab,
			                                  ivocab=self.ivocab,
			                                  word_vector_size=self.word_vector_size,
			                                  to_return=self.answer_vec))
		# NOTE: here we assume the answer is one word!
		if self.input_mask_mode == 'word':
			input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32))
		elif self.input_mask_mode == 'sentence':
			input_masks.append(
				np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32))
		else:
			raise Exception("invalid input_mask_mode")

	return inputs, questions, answers, input_masks


def neural_net_model(data):
	L1 = {'weights': tf.Variable(tf.truncated_normal([5, 5, 1, opch_L1], stddev=0.1)),
	      'biases': tf.Variable(tf.ones([opch_L1]) / 10)}

	L2 = {'weights': tf.Variable(tf.truncated_normal([5, 5, opch_L1, opch_L2], stddev=0.1)),
	      'biases': tf.Variable(tf.ones([opch_L2]) / 10)}

	L3 = {'weights': tf.Variable(tf.truncated_normal([4, 4, opch_L2, opch_L3], stddev=0.1)),
	      'biases': tf.Variable(tf.ones([opch_L3]) / 10)}

	L4 = {'weights': tf.Variable(tf.truncated_normal([7 * 7 * opch_L3, opch_L4], stddev=0.1)),
	      'biases': tf.Variable(tf.ones([opch_L4]) / 10)}

	output = {'weights': tf.Variable(tf.truncated_normal([opch_L4, classes], stddev=0.1)),
	          'biases': tf.Variable(tf.zeros([classes]) / 10)}

	# Rectified linear activation fn and softmax
	layer1 = tf.nn.relu(
		tf.add(tf.nn.conv2d(data, L1['weights'], strides=[1, 1, 1, 1], padding='SAME'), L1['biases']))

	layer2 = tf.nn.relu(
		tf.add(tf.nn.conv2d(layer1, L2['weights'], strides=[1, 2, 2, 1], padding='SAME'), L2['biases']))

	layer3 = tf.nn.relu(
		tf.add(tf.nn.conv2d(layer2, L3['weights'], strides=[1, 2, 2, 1], padding='SAME'), L3['biases']))

	layer4 = tf.nn.relu(
		tf.add(tf.matmul(tf.reshape(layer3, shape=[-1, 7 * 7 * opch_L3]), L4['weights']), L4['biases']))

	output_layer = tf.nn.softmax(tf.add(tf.matmul(layer4, output['weights']), output['biases']))

	return output_layer


def train_neural_net(X):
	Y = neural_net_model(X)

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=Y_)  # loss function
	cross_entropy = tf.reduce_mean(cross_entropy) * 100

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

*
test_data = {X: np.reshape(mnist.test.images, (-1, 28, 28, 1)), Y_: mnist.test.labels}
a, c = sessn.run([accuracy, cross_entropy], feed_dict=test_data)
print('Test Accuracy : ', a)

if __name__ == '__main__':
	train_neural_net(X)
