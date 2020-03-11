"""test example works
"""
from .context import rlmini

def test_example_error():
    """ test example model has low bias and variance"""
    from rlmini.example import example

    #get example model bias and stdev
    bias, sigma = example()

    #we know...
    #data mean = 5.12
    #and
    #data sigma = 0.59

    #assert bias and variance are acceptable
    assert (abs(bias) < .05) #about 1% of mean
    assert (sigma < .6) #on par with that of data
