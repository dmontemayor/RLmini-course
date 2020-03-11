"""example assignment"""

import numpy as np
from crpm.dataset import load_dataset
from crpm.ffn_bodyplan import read_bodyplan
from crpm.ffn_bodyplan import init_ffn
from crpm.gradientdecent import gradientdecent

def example():
    """This example demonstrates regression using a minimally deep netowrk.

    Will download example renal data with features:
        egfr, fscore2, african_ancestry,
        caucasian_ancestry, and chinese_japanese_ancestry

    Will regress egfr value using example network:
        A 2 layer network with 100 hidden logistic nodes
        and mean squared error loss function.

    Will divide observations in half for training and validation.

    Will calculate model error using the validation set.

    Returns error as bias and standard deviation values.
    """

    #Begin program
    print("")
    print("B E G I N   P R O G R A M")
    print("=========================")
    print("")

    #get the example data
    #(NOTE data is numpy array with features in rows and observations in columns)
    keys, data = load_dataset("./data/example_dataset.csv")
    nfeat = data.shape[0]
    nobv = data.shape[1]

    #diagnostic: print keys and data shape
    print(keys)
    print(data.shape)

    #partition training and validation observations
    np.random.shuffle(data.T) #np.shuffle shuffles first axis so need to shuffle transpose

    #save last half of observations for validataion
    valid = data[:, nobv//2:nobv] #pylint: disable=unsubscriptable-object

    #remove validation data
    data = data[:, 0:nobv//2] #pylint: disable=unsubscriptable-object

    #Regression TASK - regress first feature "egfr"
    #                  from the remaining features
    #                  using the example network bodyplan

    #create model from bodyplan file
    bodyplan = read_bodyplan("models/example_bodyplan.csv")

    #init model
    model = init_ffn(bodyplan)

    #train model by gradient decent with naive early stopping
    pred, cost, _ = gradientdecent(model=model,
                                   data=data[1:nfeat, :],
                                   targets=data[0, :],
                                   lossname="mse",
                                   validata=valid[1:nfeat, :],
                                   valitargets=valid[0, :],
                                   earlystop=True)

    #Report mean squared error
    print("")
    print("model MSE = " + str(cost))

    #Calcualte bias and variance
    bias = np.mean(pred-valid[0, :]) #mean of residuals
    sigma = np.std(pred-valid[0, :]) #std of residuals
    print("model bias = " + str(bias))
    print("model stdev = " + str(sigma))

    #Compare with data stats
    print("----------------------------")
    print("data mean = "+ str(np.mean(valid[0, :])))
    print("data sigma = "+ str(np.std(valid[0, :])))

    #End program
    print("")
    print("=====================")
    print("E N D   P R O G R A M")

    return bias, sigma

if __name__ == '__main__':
    example()
