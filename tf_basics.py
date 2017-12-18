import tensorflow as tf

# TF constants 
x1 = tf.constant(7)
x2 = tf.constant(2)

# Model input and output parameters
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Model params - TF graph variables
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# prod = tf.multiply(x1,x2)

# Defining a linear computational graph
linear_model = W*x+b
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# Define session to execute comp graphs
with tf.Session() as sess:
	
	# Run var init subgraph
	init = tf.global_variables_initializer()
	sess.run(init)
	
	# Training loop
	for i in range(10000):
		# Run output comp graph
		output = sess.run(train, {x: x_train, y: y_train})

 	# Final param values
	curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
	print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))	
