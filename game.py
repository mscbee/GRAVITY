import random
import numpy as np

class Gravity:
    def __init__(self, size):
        self.size = size
        block_y = 0
        block_x = random.randint(0, self.size - 1)
        basket_x = random.randint(1, self.size - 2)
        self.state = [block_y, block_x, basket_x]

    def observe(self):
        # creating a 10 x 10 grid that the neural net can use to make predictions
        canvas - [0] * self.size**2
        canvas[self.state[0] * self.size + self.state[1]] = 1
        canvas[(self.size - 1) * self.size + self.state[2] - 1] = 1
        canvas[(self.size - 1) * self.size + self.state[2] + 0] = 1
        canvas[(self.size - 1) * self.size + self.state[2] + 1] = 1
        return np.array(canvas).reshape(1, -1)

    def act(self, action):
        block_y, block_x, basket_x = self.state
        #Action takes a value of 0,1, or 2, based on whether we move left, right or stay put
        basket_x += (int(action) - 1)

        #Makes sure we don't go off screen
        basket_x = max(1, basket_x)
        basket_x = min(self.size - 2, basket_x)

        block_y += 1

        self.state = [block_y, block_x, basket_x]

        reward = 0
        if block_y == self.size - 1:
            if abs(block_x - basket_x) <= 1:
                reward = 1 #catching the block_x
            else:
                reward = -1 #missed the block
