import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import json

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

        game_over - block_y == self.size - 1

        return self.observe(), reward, game_over

    def reset(self):
        self.__init__(self.size)

if __name__ == '__main__':

    GRID_DIM = 10

    EPSILON = 0.1 # we'll explore 10% of the time and exploit the rest of the time

    LEARNING_RATE = 0.2

    LOSS_FUNCTION = "mse" # sum of mean squared error, determines how far off we are from target

    HIDDEN_LAYER1_SIZE = 100
    HIDDEN_LAYER1_ACTIVATION = "relu" # Activation function

    HIDDEN_LAYER2_SIZE = 100
    HIDDEN_LAYER2_ACTIVATION = "relu"

    BATCH_SIZE = 50 # Giving our neural net 50 examples at a time
    EPOCHS = 1000 # the extent to which we're training our network - training our model over 1000 iterations

    model = Sequential()

    # Layer 1
    model.add(
        Dense(HIDDEN_LAYER1_SIZE,
        input_shape = (GRID_DIM**2,),
        activation = HIDDEN_LAYER1_ACTIVATION
        )
    )

    # Layer 2
    model.add(
        Dense(HIDDEN_LAYER2_SIZE,
        activation = HIDDEN_LAYER2_ACTIVATION
        )
    )

    # Output layer
    model.add(Dense(3))

    model.compile(sgd(lr=LEARNING_RATE), LOSS_FUNCTION)

    env = Gravity(GRID_DIM)

    win_cnt = 0
    for epoch in range(EPOCHS):
        env.reset()
        game_over = False
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t

            if random.random()<= EPSILON:
                # Takes a random action
                action = random.randint(0,2)
            else:
                # Takes the action our neural net tells us is best
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            input_t, reward, game_over = env.act(action)
            if reward == 1:
                win_cnt += 1

        print("Epoch: {:06d}/{:06d} | Win count {}".format(epoch, EPOCHS, win_cnt))

        # Save model weights - the 'knowledge' accumulated by the network
        model.save_weights("model.h5", overwrite=True)

        with open("modle.json", "w") as outfile:
            json.dump(model.to_json(), outfile)
