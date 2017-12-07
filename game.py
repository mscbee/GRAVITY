import random

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
