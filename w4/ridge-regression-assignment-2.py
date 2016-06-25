############################
#
# Contains several functions that are used to perform gradient descent for ridge regression utilizing numpy
# i.e. the inner-workings of a package such as scikit-learn or graphlab-create for ridge regression proving I know math
#
##############################

import graphlab as gl
import numpy as np

sales = gl.SFrame('kc_house_data.gl')

def get_numpy_data(data_sframe, features, output):
    # create a constant column for the sframe, i.e "b" in y = b + mx
    data_sframe['constant'] = 1
    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features
    # select columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
    features_sframe = data_sframe[features]
    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = features_sframe.to_numpy()
    # assign the column of data_sframe associated with the output to the SArray output_sarray
    output_sarray = data_sframe[output]
    # the following will convert the SArray into a numpy array by first converting it to a list
    output_array = output_sarray.to_numpy()

    return feature_matrix, output_array

def predict_output(feature_matrix, weights):
    # assume feature_matrix is numpy matrix containing features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)

    return predictions


def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    # If feature_is_constant is True, derivative is twice the dot product of errors and feature
    if feature_is_constant:
        derivative = 2 * np.dot(errors, feature)
    # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
    else:
        derivative = 2 * np.dot(errors, feature) + 2 * l2_penalty * weight

    return derivative


def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty,
                                      max_iterations=100):
    weights = np.array(initial_weights)  # make sure it's a numpy array
    iteration = 1
    while iteration <= 100:
        # Compute predictions
        predictions = predict_output(feature_matrix, weights)

        # Compute errors
        errors = predictions - output

        for i in xrange(len(weights)):  # loop over each weight
            # Compute derivative using feature_derivative_ridge, ensuring we calculate constant correctly
            derivative = feature_derivative_ridge(errors, feature_matrix[:, i], l2_penalty, True if i == 0 else False)

            # Subtract the step size times the derivative from the current weight
            weights[i] -= (step_size * derivative)

        # All weights are updated. Time for the next iteration
        iteration += 1

    return weights

def get_rss(feature_matrix, weights, output):
    predictions = predict_output(feature_matrix, weights)
    residuals = predictions - output
    RSS = sum(map(lambda x: x**2, residuals))

    return RSS



