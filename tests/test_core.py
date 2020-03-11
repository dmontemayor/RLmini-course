"""test crpm package is installed properly
"""
from .context import rlmini

def test_load_dataset():
    """ test load_dataset function from crpm package runs"""
    from crpm.dataset import load_dataset
    keys, data = load_dataset("./data/example_dataset.csv")
    assert(keys == ["egfr", "fscore2", "african_ancestry", "caucasian_ancestry", "chinese_japanese_ancestry"])
