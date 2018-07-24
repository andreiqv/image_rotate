#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
in 
- conv 5х5 - pool 3х3 
- conv 4х4 - pool 3х3  
- conv 3х3 - pool 2х2  
- reshape - 1024 - dense -dense - mse

Validation:
train: 296.32 - 194.00
train: 326.57 - 356.00
train: 378.93 - 156.00
valid: 75.48 - 76.00
valid: 239.89 - 51.00
iteration 420: train_acc=0.2516, valid_acc=0.2653

"""

# export CUDA_VISIBLE_DEVICES=1

from __future__ import absolute_import,  division, print_function
import tensorflow as tf
import sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)

#import load_data
import _pickle as pickle
import gzip

BATCH_SIZE = 10
NUM_ITERS = 500000

data_file = "dump.gz"
f = gzip.open(data_file, 'rb')
data = pickle.load(f)
#data_1 = load_data(in_dir, img_size=(540,540))
#data = split_data(data1, ratio=(6,1,3))

train = data['train']
valid = data['valid']
test  = data['test']
print('train size:', train['size'])
print('valid size:', valid['size'])
print('test size:', test['size'])
im0 = train['images'][0]
print('Data was loaded.')
print(im0.shape)
#sys.exit()

#train['images'] = [np.transpose(t) for t in train['images']]
#valid['images'] = [np.transpose(t) for t in valid['images']]
#test['images'] = [np.transpose(t) for t in test['images']]
num_train_batches = train['size'] // BATCH_SIZE
num_valid_batches = valid['size'] // BATCH_SIZE
num_test_batches = test['size'] // BATCH_SIZE
print('num_train_batches:', num_train_batches)
print('num_valid_batches:', num_valid_batches)
print('num_test_batches:', num_test_batches)

SAMPLE_SIZE = train['size']

# some functions

def weight_variable(shape, name=None):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W, name=None):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)

def max_pool_2x2(x, name=None):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME',  name=name) 

def max_pool_3x3(x, name=None):
	return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME',  name=name)


def convolutionLayer(p_in, kernel, pool_size, num_in, num_out, func=None, name=''):
	W = weight_variable([kernel[0], kernel[1], num_in, num_out], name='W'+name)  # 32 features, 5x5
	b = bias_variable([num_out], name='b'+name)
	if func:
		h = func(conv2d(p_in, W, name='conv'+name) + b, name='relu'+name)
	else:
		h = conv2d(p_in, W, name='conv'+name) + b
	if pool_size == 2:
		p_out = max_pool_2x2(h, name='pool'+name)
	elif pool_size == 3:
		p_out = max_pool_3x3(h, name='pool'+name)
	else:
		raise("bad pool size")
	print('p{0} = {1}'.format(name, p_out))
	return p_out


# Create a new graph
graph = tf.Graph() # no necessiry

with graph.as_default():

	# 1. Construct a graph representing the model.
	x = tf.placeholder(tf.float32, [None, 540, 540]) # Placeholder for input.
	y = tf.placeholder(tf.float32, [None])   # Placeholder for labels.
	
	x_image = tf.reshape(x, [-1,540,540,1])


	# 1: conv layer 1
	#num_filters_1 = 16
	p1 = convolutionLayer(x_image, kernel=(5,5), pool_size=3, num_in=1, num_out=16, 
		func=tf.nn.relu, name='1') # 180 x 180
	p2 = convolutionLayer(p1, kernel=(5,5), pool_size=3, num_in=16, num_out=16, 
		func=tf.nn.relu, name='2')  # 60 x 60 
	p3 = convolutionLayer(p2, kernel=(4,4), pool_size=3, num_in=16, num_out=32, 
		func=tf.nn.relu, name='3')   # 20 x 20 
	p4 = convolutionLayer(p3, kernel=(4,4), pool_size=2, num_in=32, num_out=32, 
		func=tf.nn.relu, name='4')   # 10 x 10 
	p5 = convolutionLayer(p4, kernel=(3,3), pool_size=2, num_in=32, num_out=32, 
		func=tf.nn.relu, name='5')   # 5 x 5

	"""
	W1 = weight_variable([5, 5, 1, num_filters_1], name='W1')  # 32 features, 5x5
	b1 = bias_variable([num_filters_1], name='b1')
	h1 = tf.nn.relu(conv2d(x_image, W1, name='conv1') + b1, name='relu1')
	p1 = max_pool_3x3(h1, name='pool1')
	print('p1 =', p1)
	# 180 x 180
		
	# 2: conv layer 2

	num_filters_2 = 16
	W2 = weight_variable([4, 4, num_filters_1, num_filters_2], name='W2')  
	b2 = bias_variable([num_filters_2], name='b2')
	h2 = tf.nn.relu(conv2d(p1, W2, name='conv2') + b2, name='relu2')
	p2 = max_pool_3x3(h2, name='pool2')
	print('p2 =', p2)   
	# 60 x 60 

	# 3: conv layer 3
	num_filters_3 = 32
	W3 = weight_variable([3, 3, num_filters_2, num_filters_3], name='W3')  
	b3 = bias_variable([num_filters_3], name='b3')
	h3 = tf.nn.relu(conv2d(p2, W3, name='conv3') + b3, name='relu3')
	p3 = max_pool_2x2(h3, name='pool3')
	print('p3 =', p3)   
	# 30 x 30   

	# 4: conv layer 4
	num_filters_4 = 32
	W4 = weight_variable([3, 3, num_filters_3, num_filters_4], name='W4')  
	b4 = bias_variable([num_filters_4], name='b4')
	h4 = tf.nn.relu(conv2d(p3, W4, name='conv4') + b4, name='relu4')
	p4 = max_pool_2x2(h4, name='pool4')
	print('p4 =', p4)   
	# 15 x 15
	"""


	# 5: fully-connected layert
	num_filters = 32

	num_neurons_5 = 1024
	p4_flat = tf.reshape(p5, [-1, 5*5*num_filters])
	W5 = weight_variable([5*5*num_filters, num_neurons_5], name='W5')
	b5 = bias_variable([num_neurons_5], name='b5')
	h5 = tf.nn.relu(tf.matmul(p4_flat, W5) + b5, name='relu5')
	print('h5 =', h5)

	# 6: fully-connected layert
	num_neurons_6 = 128
	W6 = weight_variable([num_neurons_5, num_neurons_6], name='W6')
	b6 = bias_variable([num_neurons_6], name='b6')
	h6 = tf.nn.relu(tf.matmul(h5, W6) + b6, name='relu6')
	#h5 = tf.matmul(h4, W5) + b5
	print('h6 =', h6)

	# 7: output layer
	num_neurons_7 = 1
	W7 = weight_variable([num_neurons_6, num_neurons_7], name='W7')
	b7 = bias_variable([num_neurons_7], name='b7')
	#h6 = tf.nn.relu(tf.matmul(h5, W6) + b6)
	h7 = tf.matmul(h6, W7) + b7

	output = h7
	print('output =', output)

	# 2. Add nodes that represent the optimization algorithm.

	loss = tf.reduce_mean(tf.square(output - y))
	#loss = tf.reduce_mean(tf.squared_difference(y, output))
	#loss = tf.nn.l2_loss(output - y)
	#loss = tf.losses.mean_squared_error(labels=y, predictions=output)
	
	train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
	#train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
		
	#loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
	#train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
	#correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# 3. Execute the graph on batches of input data.
	with tf.Session() as sess:  # Connect to the TF runtime.
		init = tf.global_variables_initializer()
		sess.run(init)	# Randomly initialize weights.
		for iteration in range(NUM_ITERS):			  # Train iteratively for NUM_iterationS.		 

			if iteration % 50 == 0:			  

				#print('Validation:')
				output_values = output.eval(feed_dict = {x:train['images'][:3]})
				#print('train: {0:.2f} - {1:.2f}'.format(output_values[0][0]*360, train['labels'][0]*360))
				#print('train: {0:.2f} - {1:.2f}'.format(output_values[1][0]*360, train['labels'][1]*360))
				output_values = output.eval(feed_dict = {x:valid['images'][:2]})
				#print('valid: {0:.2f} - {1:.2f}'.format(output_values[0][0]*360, valid['labels'][0]*360))
				#print('valid: {0:.2f} - {1:.2f}'.format(output_values[1][0]*360, valid['labels'][1]*360))

				train_accuracy = loss.eval(feed_dict = {x:train['images'][0:BATCH_SIZE], y:train['labels'][0:BATCH_SIZE]})
				valid_accuracy = loss.eval(feed_dict = {x:valid['images'][0:BATCH_SIZE], y:valid['labels'][0:BATCH_SIZE]})
			

				train_accuracy = np.mean( [loss.eval( \
					feed_dict={x:train['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
					y:train['labels'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
					for i in range(0,num_train_batches)])
				valid_accuracy = np.mean([ loss.eval( \
					feed_dict={x:valid['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
					y:valid['labels'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
					for i in range(0,num_valid_batches)])

				print('iteration {0:3}: train_acc={1:0.4f}, valid_acc={2:0.4f}'.\
					format(iteration, train_accuracy, valid_accuracy))

				"""
				train_accuracy = accuracy.eval(
					feed_dict={x:x_train[:SAMPLE_SIZE], y:y_train[:SAMPLE_SIZE]})
				validation_accuracy = accuracy.eval(
					feed_dict={x:x_valid[:SAMPLE_SIZE], y:y_valid[:SAMPLE_SIZE]})
				#x_valid_1000, y_valid_1000 = mnist.validation.next_batch(size)
				print('iteration {0:3}: train_acc={1:0.3f}, valid_acc={2:0.3f}'.format(iteration, train_accuracy, validation_accuracy))
			
				# sess.run(b2)  
				#print(accuracy.eval(feed_dict = {x : mnist.test.images[0:10], y : mnist.test.labels[0:10]}))
				"""
			
			a1 = iteration*BATCH_SIZE % train['size']
			a2 = (iteration + 1)*BATCH_SIZE % train['size']
			x_data = train['images'][a1:a2]
			y_data = train['labels'][a1:a2]
			if len(x_data) <= 0: continue
			sess.run(train_op, {x: x_data, y: y_data})  # Perform one training iteration.		
			#print(a1, a2, y_data)			

		# Save the comp. graph

		x_data, y_data =  valid['images'], valid['labels'] #mnist.train.next_batch(BATCH_SIZE)		
		writer = tf.summary.FileWriter("output", sess.graph)
		print(sess.run(train_op, {x: x_data, y: y_data}))
		writer.close()  

		# Test of model
		#test_accuracy = loss.eval(feed_dict={x:test['images'][0:BATCH_SIZE], y:test['labels'][0:BATCH_SIZE]})
		num_test_batches = test['size'] // BATCH_SIZE
		test_accuracy = np.mean([ loss.eval( \
			feed_dict={x:test['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
			y:test['labels'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
			for i in range(num_test_batches) ])
		print('Test of model')
		print('Test_accuracy={0:0.4f}'.format(test_accuracy))

		"""
		# Inference
		batch = mnist.test.next_batch(BATCH_SIZE)
		softmax = tf.nn.softmax(logits)
		output = softmax.eval(feed_dict = {x:batch[0]})
		predict = [np.argmax(output[i]) for i in range(BATCH_SIZE)] 
		target = [np.argmax(batch[1][i]) for i in range(BATCH_SIZE)]

		for i in range(BATCH_SIZE):
			print('{0} -> {1} -- {2}'.format(output[i], predict[i], target[i]))
		
		# Saver
		saver = tf.train.Saver()		
		saver.save(sess, './save_model/my_test_model')  
		"""


