from __future__ import print_function
import math
import sys

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt 
from sklearn import metrics
import numpy as np 
import pandas as pd 
import tensorflow as tf 
Dataset = tf.data.Dataset


def train_model(learning_rate, steps, batch_size, input_feature):
	periods = 10
	steps_per_period = steps / periods

	target_label = "median_house_value"
	# specify the feature(s) -- [This is the input value]
	feature_1 = california_housing_dataframe[[input_feature]]

	# give the feature a label
	feature_columns = [tf.feature_column.numeric_column(input_feature)]

	# specify the target value(s) -- [This is the expected output value]
	targets = california_housing_dataframe[target_label]

	# Create the input function
	training_input_fn = lambda:input_fn_1(feature_1, targets, batch_size = batch_size)
	prediction_input_fn = lambda:input_fn_1(feature_1, targets, num_epochs = 1, shuffle = False)


	# setup the gradient descent with the learn_rate
	my_optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
	# clipping the gradient just caps them
	my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)



	# Setup the linear regressor -- 
	# with the previously made optimizer with the specified learning rate
	linear_regressor = tf.estimator.LinearRegressor(
		feature_columns = feature_columns,
		optimizer = my_optimizer
	)

	# Set up to plot the state of our model's line each period.
	plt.figure(figsize = (15, 6))
	plt.subplot(1, 2, 1)
	plt.title("Learned Line by Period")
	plt.ylabel(target_label)
	plt.xlabel(input_feature)
	sample = california_housing_dataframe.sample(n = 300)
	plt.scatter(sample[input_feature], sample[target_label])
	colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]


	# Train the model inside a loop so we can periodically assess
	print("Training model...")
	print("RMSE (on training data): ")
	root_mean_squared_errors = [] # our list of errors
	for period in range(0, periods):
		# Train the model starting from the prior state 
		linear_regressor.train(
			input_fn = training_input_fn,
			steps = steps_per_period
		)

		# Compute Predicitons
		predictions = linear_regressor.predict(input_fn = prediction_input_fn)
		predictions = np.array([item['predictions'][0] for item in predictions])


		# Compute the Loss
		root_mean_squared_error = math.sqrt(metrics.mean_squared_error(predictions, targets))


		# Print the Loss -- So we see whats going on
		print("		period %02d: %0.2f" % (period, root_mean_squared_error))


		# Add the loss metrics from this period to our list. 
		root_mean_squared_errors.append(root_mean_squared_error)

		# Finally, track the weights and biases over time.
		# Make sure that the data and line are plotted neatly
		y_extents = np.array([0, sample[target_label].max()])

		weight = linear_regressor.get_variable_value('linear/linear_model/%s/weight' % input_feature)[0]
		bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

		x_extents = (y_extents - bias) / weight

		x_extents = np.maximum(np.minimum(x_extents, sample[input_feature].max()), sample[input_feature].min() )

		y_extents = weight * x_extents + bias

		plt.plot(x_extents, y_extents, color = colors[period])
	print("Training Finished!!!")

	# Output a graph of loss metrics over periods. 
	plt.subplot(1, 2, 2)
	plt.ylabel('RMSE')
	plt.xlabel('Periods')
	plt.title("Root Mean Squared Error vs. Periods")
	plt.tight_layout()
	plt.plot(root_mean_squared_errors)


	# Output a table with calibration data
	calibration_data = pd.DataFrame()
	calibration_data["predictions"] = pd.Series(predictions)
	calibration_data["targets"] = pd.Series(targets)
	display.display(calibration_data.describe())

	print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)

def input_fn_1(features, targets, batch_size = 1, shuffle = True, num_epochs = None):
	"""
	Trains a linear regression model with one feature
	"""
	# convert panda's data to numpy arrays
	features = {key:np.array(value) for key,value in dict(features).items()}


	# construct a dataset, and configure batching/repeating.
	ds = Dataset.from_tensor_slices((features,targets)) # 2GB limit
	ds = ds.batch(batch_size).repeat(num_epochs)

	# Shuffle data if needed
	if shuffle:
		ds = ds.shuffle(buffer_size = 10000)


	# Return the next batch of data
	features, labels = ds.make_one_shot_iterator().get_next()
	return features, labels

learn_rate = 0.0000001
total_steps = 100
feature = "total_rooms"
batch_size = 1

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


# load data
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv",sep = ",")


# randomize dataset so that the gradient descent performance isn't hurt by orderings
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe.describe()

'''
TRAIN MODEL HERE
'''
train_model(learning_rate = learn_rate, steps = total_steps, batch_size = batch_size, input_feature = feature)















