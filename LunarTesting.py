import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt

EPISODES = 40000
champion = "Current_Champion_1.h5"
discount_factor = 0.99
hidden_layer1 = 40
hidden_layer2 = 40
learning_rate = 5E-5

use_max_action = True
render = False

env = gym.make('LunarLander-v2')
org_state_size = env.observation_space.shape[0]
state_size = 2 * org_state_size
action_size = env.action_space.n


def model_construct(s_size, a_size, hidden1, hidden2, lr):
    model = Sequential()
    model.add(Dense(hidden1, input_dim=s_size, activation='relu', kernel_initializer='glorot_uniform'))
    # model.add(BatchNormalization())
    model.add(Dense(hidden2, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(a_size, activation='softmax', kernel_initializer='glorot_uniform'))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=lr))
    return model


def action_selection(m, s, a_size):
    policy = m.predict(s, batch_size=1).flatten()
    # print policy
    # print np.where(policy == policy.max())[0][0]
    # print np.random.choice(a_size, 1, p=policy)[0]
    if use_max_action:
        return np.where(policy == policy.max())[0][0]
    else:
        return np.random.choice(a_size, 1, p=policy)[0]


model = model_construct(state_size, action_size, hidden_layer1, hidden_layer2, learning_rate)
model.load_weights(champion)

scores = []
for episode in range(EPISODES):
    done = False
    score = 0
    state = env.reset()
    state = np.reshape(state, [1, org_state_size])
    previous_state = np.zeros((1, org_state_size))

    while not done:

        if render:
            env.render()

        state_holder = np.zeros((1, state_size))
        state_holder[0, 0:8] = previous_state[0]
        state_holder[0, 8::] = state[0]

        state = state_holder

        # get action for the current state and go one step in environment
        action = action_selection(model, state, action_size)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, org_state_size])

        previous_state = state[:, -org_state_size::]
        score += reward
        state = next_state

        if done:
            scores.append(score)
            print "Episode: {0}   score: {1}".format(episode, score)

print "====Mean Score of {0} Trials====".format(EPISODES), np.mean(scores)
print "===Minimum Score of {0} Trials=====".format(EPISODES), np.min(scores)
print "====Maximum Score of {0} Trials====".format(EPISODES), np.max(scores)
print "====Standard Deviation of Scores of {0} Trials====".format(EPISODES), np.std(scores)

plt.hist(scores, bins=100)
plt.show()
