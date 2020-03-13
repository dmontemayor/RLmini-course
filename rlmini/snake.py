"""Snake Game"""

import random
import curses

def snakegame():
    """Snake Game rendered with ascii characters
    """
    #curses to initialize screen
    screen = curses.initscr()

    #set initial curser to zero so doesn't show up on screen
    curses.curs_set(0)

    #get screen width and height
    scrh, scrw = screen.getmaxyx()

    #create a new window using width and height and starting at top-left corner
    window = curses.newwin(scrh, scrw, 0, 0)

    #set to accept keypad input
    window.keypad(1)

    #refresh every 100 ms
    window.timeout(100)

    #set snake initial position in middle of screen
    snake_x = scrw//2
    snake_y = scrh//2

    #create snake initial body segments (3 in x direction)
    snake = [[snake_y, snake_x],
             [snake_y, snake_x-1],
             [snake_y, snake_x-2]]

    #have initial key entry be to the right so snake doesn't hit its self
    key = curses.KEY_RIGHT

    #initial food some where behind the snake
    food = [scrh//4, scrw//4]

    #add the diamond character on the window to represent the food
    window.addch(food[0], food[1], curses.ACS_DIAMOND)

    #init snake alive with zero points
    alive = True
    score = 0

    #main game loop
    while alive:

        window.refresh()
        #display score
        window.addstr(0, 0, str(score))

        #get action input
        next_key = window.getch()
        #Note: next_key will = -1 if no key is pressed

        #keep moving in same direction unless something was pressed
        key = key if next_key == -1 else next_key

        #update snake position
        new_head = [snake[0][0], snake[0][1]]
        if key == curses.KEY_DOWN:
            new_head[0] += 1
        if key == curses.KEY_UP:
            new_head[0] -= 1
        if key == curses.KEY_RIGHT:
            new_head[1] += 1
        if key == curses.KEY_LEFT:
            new_head[1] -= 1

        #stack new head on snake
        snake.insert(0, new_head)

        #check if snake head ate food
        if snake[0] == food:
            #increment score
            score += 1
            #create new food at random position at least one space away from walls
            food = None
            while food is None:
                newfood = [random.randint(1, scrh-1), random.randint(1, scrw-1)]
                #update food position if new food position is not in snakes body
                food = newfood if newfood not in snake else None
            #add new food character to window
            window.addch(food[0], food[1], curses.ACS_DIAMOND)
        else:
            #pop off snake's tail
            tail = snake.pop()
            #add a space character where the tail was to erase from window
            window.addch(tail[0], tail[1], ' ')

        #add head of snake to the window as checkerboard character
        window.addch(snake[0][0], snake[0][1], curses.ACS_CKBOARD)

        #check to see if snake deaded
        #head of snake's y position hits bottom or top of screen
        if snake[0][0] in [0, scrh]:
            alive = False
        #head of snake's x position hits left or right of screen
        if snake[0][1] in [0, scrw]:
            alive = False
        #snake's head hits anywhere on snakes body
        if snake[0] in snake[1:]:
            alive = False

    #end main game loop

    #close window and return score
    curses.endwin()
    return score

if __name__ == '__main__':
    snakegame()
