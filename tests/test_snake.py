"""test snake game
"""
from .context import rlmini

#this test doesn't work because of curses
def bad_test_snake_score():
    """ test snake game will end and return 0 if do nothing"""
    from rlmini.snake import snakegame

    #get snake return
    score = snakegame()

    #should be true if point is initially behind the snake
    assert (score == 0)
