import sys
sys.path.insert(0,'../Data')
from data_cleaner import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA

# # %% Let's create some toy data
# # Turn interactive mode on
# plt.ion()
# n_observations = 100
# # return a subplot axes at the given grid position on the "canvas"
# fig, ax = plt.subplots(1, 1)
# # x axis is from -3 to 3 with n_observations evenly spaced
# xs = np.linspace(-3, 3, n_observations)

# print xs

# print type(xs)

input_string_array = np.array(usable_data)
input_float_array = input_string_array.astype(float)
# sanity_check_array = np.array(cleaned_data)

# Randomize input
random_array = np.random.permutation(input_float_array)

# Constructing y val of array from violent crime rates
y_array = random_array[:, -1]
# Constructing x val of array but splicing of last element in every subarray
x_array = random_array[:, :-1]
# Reduce dimensionality using Principal Component Analysis
# print "Number of features prior to dimensionality reduction"
# print len(x_array[0])
pca = PCA(.90)
x_reduced = pca.fit_transform(x_array)
# print "Number of features post dimensionality reduction"
# print len(x_reduced[0])
#print pca.components_
# print "Variance Ratio"
# print pca.explained_variance_ratio_

#randomize input



training_batch_size = int(len(random_array)*.8)

x_training_data = x_array[:training_batch_size]
y_training_data = y_array[:training_batch_size]


x_testing_data = x_array[training_batch_size:]
y_testing_data = y_array[training_batch_size:]



n = len(x_testing_data[0])
# %% tf.placeholders for the input and output of the network. Placeholders are
# variables which we need to fill in when we are ready to compute the graph.
X = tf.placeholder(tf.float32, [n,1])
# Try intializing with random normal an zeros
W = tf.Variable(tf.random_normal([1, n]))
b = tf.Variable(tf.random_normal([1]))
Y = tf.placeholder(tf.float32)
Y_prediction = tf.Variable(tf.random_normal([1]), name='bias')
Y_prediction = tf.add(tf.matmul(W, X), b)

# %% Loss function will measure the distance between our observations
# and predictions and average over them.
cost = tf.reduce_sum(tf.pow(Y_prediction - Y, 2)) / (training_batch_size - 1)


# %% if we wanted to add regularization, we could add other terms to the cost,
# e.g. ridge regression has a parameter controlling the amount of shrinkage
# over the norm of activations. the larger the shrinkage, the more robust
# to collinearity.
# cost = tf.add(cost, tf.mul(1e-6, tf.global_norm([W])))

# %% Use gradient descent to optimize W,b
# Performs a single step in the negative gradient
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

n_epochs = 1000

with tf.Session() as sess:
	# Here we tell tensorflow that we want to initialize all
    # the variables in the graph so we can use them
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    prev_training_cost = 0.0
    for epoch_i in range(n_epochs):

    	training_cost = 0
    	for index in range(training_batch_size):

    		x_reshaped = np.array(x_training_data[index]).reshape(n,1)
    		sess.run(optimizer, feed_dict={X:x_reshaped, Y:y_training_data[index]})
    		holder_cost = sess.run(cost, feed_dict={X: x_reshaped, Y: y_training_data[index]})
    		# print holder_cost
    		training_cost += holder_cost
        print(training_cost)

        # Allow the training to quit if we've reached a minimum
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost = training_cost

# print "Cleaned_data"
# print sanity_check_array
# # print "OG data"
# # print usable_data[:10]
# print "Float Data"
# print input_float_array

# print "x Values_________________________________________________________________________"
# print x_array

# print "y Values_________________________________________________________________________"
# print y_array

# print y_array.size
