import tensorflow as tf

x1 = tf.constant(7)
x2 = tf.constant(2)

prod = tf.multiply(x1,x2)

with tf.Session() as sess:
	output = sess.run(prod)

print(output)
