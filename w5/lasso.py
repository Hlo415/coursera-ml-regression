import graphlab
import numpy as np

sales = graphlab.SFrame('kc_house_data.gl/')

# In the dataset, 'floors' was defined with type string. Convert to int.
sales['floors'] = sales['floors'].astype(int)


def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1  # this is how you add a constant column to an SFrame
    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features  # this is how you combine two lists
    # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
    features_sframe = data_sframe[features]
    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = features_sframe.to_numpy()
    # assign the column of data_sframe associated with the output to the SArray output_sarray
    output_sarray = data_sframe[output]
    # the following will convert the SArray into a numpy array by first converting it to a list
    output_array = output_sarray.to_numpy()

    return (feature_matrix, output_array)

def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)

    return(predictions)

def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    return feature_matrix / norms, norms

def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = predict_output(feature_matrix, weights)

    ro_i = np.sum(feature_matrix[:, i] * (output - prediction + weights[i] * feature_matrix[:, i]))

    if i == 0:  # intercept -- do not regularize
        new_weight_i = ro_i

    elif ro_i < -l1_penalty / 2.:
        new_weight_i = ro_i + l1_penalty / 2

    elif ro_i > l1_penalty / 2.:
        new_weight_i = ro_i - l1_penalty / 2

    else:
        new_weight_i = 0.

    return new_weight_i

def lasso_cyclical_coordinate_descent(feature_matrix, output, weights, l1_penalty, tolerance):
    weight_diff = []

    for i in range(len(weights)):
        old_weight_i = weights[i]
        weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
        weight_diff.append(abs(old_weight_i - weights[i]))

    if max(weight_diff) > tolerance:
        lasso_cyclical_coordinate_descent(feature_matrix, output, weights, l1_penalty, tolerance)
    else:
        return weights

def get_rss(feature_matrix, weights, output):
    predictions = predict_output(feature_matrix, weights)
    residuals = predictions - output
    RSS = sum(map(lambda x: x**2, residuals))

    return RSS


def get_nnz(weights, features):
    # Will return the names of features with non-zero weights
    nnz = {}
    nnz['intercept'] = weights[0]

    for i in range(1, len(features) + 1):
        if weights[i] != 0:
            nnz[features[i - 1]] = weights[i]

    return nnz

