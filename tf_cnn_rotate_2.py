#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
in 
- conv 5х5 - pool 3х3 
- conv 4х4 - pool 3х3  
- conv 3х3 - pool 2х2  
- reshape - 1024 - dense -dense - mse
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


train['images'] = [np.transpose(t) for t in train['images']]
valid['images'] = [np.transpose(t) for t in valid['images']]
test['images'] = [np.transpose(t) for t in test['images']]


BATCH_SIZE = 5
SAMPLE_SIZE = train['size']
NUM_STEPS = 1000

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

# Create a new graph
graph = tf.Graph() # no necessiry

with graph.as_default():

	# 1. Construct a graph representing the model.
	x = tf.placeholder(tf.float32, [None, 540, 540]) # Placeholder for input.
	y = tf.placeholder(tf.float32, [None])   # Placeholder for labels.
	
	x_image = tf.reshape(x, [-1,540,540,1])

	# 1: conv layer 1
	num_filters_1 = 8
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
	num_filters_3 = 16
	W3 = weight_variable([3, 3, num_filters_2, num_filters_3], name='W3')  
	b3 = bias_variable([num_filters_3], name='b3')
	h3 = tf.nn.relu(conv2d(p2, W3, name='conv3') + b3, name='relu3')
	p3 = max_pool_2x2(h3, name='pool3')
	print('p3 =', p3)   
	# 30 x 30   

	# 4: fully-connected layert
	num_neurons_4 = 1024
	p3_flat = tf.reshape(p3, [-1, 30*30*num_filters_3])
	W4 = weight_variable([30*30*num_filters_3, num_neurons_4], name='W4')
	b4 = bias_variable([num_neurons_4], name='b4')
	h4 = tf.nn.relu(tf.matmul(p3_flat, W4) + b4, name='relu4')
	print('h4 =', h4)

	# 5: fully-connected layert
	num_neurons_5 = 1
	W5 = weight_variable([num_neurons_4, num_neurons_5], name='W5')
	b5 = bias_variable([num_neurons_5], name='b5')
	#h5 = tf.nn.relu(tf.matmul(h4, W5) + b5, name='relu5')
	h5 = tf.matmul(h4, W5) + b5
	print('h5 =', h5)

	# 6: output layer
	"""
	num_neurons_6 = 1
	W6 = weight_variable([num_neurons_5, num_neurons_6], name='W6')
	b6 = bias_variable([num_neurons_6], name='b6')
	h6 = tf.nn.relu(tf.matmul(h5, W6) + b6)
	"""

	output = h5
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
		for step in range(NUM_STEPS):			  # Train iteratively for NUM_STEPS.		 
			#x_data, y_data = mnist.train.next_batch(BATCH_SIZE) # Load one batch of input data
			
			a1 = step*BATCH_SIZE % train['size']
			a2 = (step+1)*BATCH_SIZE % train['size']
			x_data = train['images'][a1:a2]
			y_data = train['labels'][a1:a2]
			if len(x_data) <= 0: continue
			#print(a1, a2, y_data)			

			if step % 5 == 0:			  

				print('Validation:')
				output_values = output.eval(feed_dict = {x:train['images']})
				print('train: {0:.2f} - {1:.2f}'.format(output_values[0][0]*360, train['labels'][0]*360))
				print('train: {0:.2f} - {1:.2f}'.format(output_values[1][0]*360, train['labels'][1]*360))
				print('train: {0:.2f} - {1:.2f}'.format(output_values[2][0]*360, train['labels'][2]*360))
				output_values = output.eval(feed_dict = {x:valid['images']})
				print('valid: {0:.2f} - {1:.2f}'.format(output_values[0][0]*360, valid['labels'][0]*360))
				print('valid: {0:.2f} - {1:.2f}'.format(output_values[1][0]*360, valid['labels'][1]*360))

				train_accuracy = loss.eval(feed_dict = {x:train['images'], y:train['labels']})
				valid_accuracy = loss.eval(feed_dict = {x:valid['images'], y:valid['labels']})
				print('step {0:3}: train_acc={1:0.4f}, valid_acc={2:0.4f}\n'.\
					format(step, train_accuracy, valid_accuracy))

				"""
				train_accuracy = accuracy.eval(
					feed_dict={x:x_train[:SAMPLE_SIZE], y:y_train[:SAMPLE_SIZE]})
				validation_accuracy = accuracy.eval(
					feed_dict={x:x_valid[:SAMPLE_SIZE], y:y_valid[:SAMPLE_SIZE]})
				#x_valid_1000, y_valid_1000 = mnist.validation.next_batch(size)
				print('step {0:3}: train_acc={1:0.3f}, valid_acc={2:0.3f}'.format(step, train_accuracy, validation_accuracy))
			
				# sess.run(b2)  
				#print(accuracy.eval(feed_dict = {x : mnist.test.images[0:10], y : mnist.test.labels[0:10]}))
				"""

			sess.run(train_op, {x: x_data, y: y_data})  # Perform one training step.		

		# Save the comp. graph

		x_data, y_data =  valid['images'], valid['labels'] #mnist.train.next_batch(BATCH_SIZE)		
		writer = tf.summary.FileWriter("output", sess.graph)
		print(sess.run(train_op, {x: x_data, y: y_data}))
		writer.close()  

		"""  
		# Test of model
		test_accuracy = accuracy.eval(feed_dict={x:x_test, y:y_test})
		print('Test_accuracy={0:0.4f}'.format(test_accuracy))

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




"""
Possible errors:

ValueError: Only call `softmax_cross_entropy_with_logits` with named arguments
- 
Use `tf.global_variables_initializer` instead.

"""