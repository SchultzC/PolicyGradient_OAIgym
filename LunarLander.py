import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import pandas as pd

# Create the Lunar Lander V2 openAI gym environment
env = gym.make('LunarLander-v2')

# Determine the length of the state variable information and the size of the action space
org_state_size = env.observation_space.shape[0]
state_size = 2 * org_state_size
action_size = env.action_space.n

# Set the number of episodes an agent can run
EPISODES = 80001

# Hyper Parameters
batch_size = 32
learning_rate = 5E-5/batch_size
discount_factor = 0.99
hidden_layer1 = 64
hidden_layer2 = 64
state_array = []
action_array = []
reward_array = []


# Function to construct the neural network and initialize the weights
def model_construct(s_size, a_size, hidden1, hidden2, lr):
    model = Sequential()
    model.add(Dense(hidden1, input_dim=s_size, activation='relu', kernel_initializer='glorot_uniform'))
    # model.add(BatchNormalization())
    model.add(Dense(hidden2, activation='relu', kernel_initializer='glorot_uniform'))
    # model.add(BatchNormalization())
    model.add(Dense(a_size, activation='softmax', kernel_initializer='glorot_uniform'))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=lr))
    return model


# Retrieve an action given the current network weights
def action_selection(m, s, a_size):

    # Obtain a distribution over actions
    action_prob_distribution = m.predict(s, batch_size=1).flatten()

    # Choose an action from this distribution
    return np.random.choice(a_size, 1, p=action_prob_distribution)[0]


# Train the model given the states, actions and rewards for the episode
def train_model(m, sts, acts, rwds, gamma, s_size, a_size):

    episode_length = len(sts)

    discounted_rewards = np.zeros_like(rwds)
    discounted_sum = 0

    # Calculate the discounted sum of rewards back through the trajectory
    for t in reversed(range(0, len(rwds))):
        discounted_sum = discounted_sum * gamma + rwds[t]
        discounted_rewards[t] = discounted_sum

    # Normalize the discounted rewards to be 0 mean unit variance
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    update_inputs = np.zeros((episode_length, s_size))
    advantages = np.zeros((episode_length, a_size))

    # Calculate the advantage
    for i in range(episode_length):
        update_inputs[i] = sts[i]
        advantages[i][acts[i]] = discounted_rewards[i]  # Actor-Critic would subtract a value estimate baseline here

    # Fit the network model given the states and the Advantages
    m.fit(update_inputs, advantages, epochs=1, verbose=0)

    # return three empty arrays to reset the states, actions and rewards.
    # The previous data was used only once to help compute gradient for one episode, then discarded.
    return [], [], []


# Build the model and initialize the weights
model = model_construct(state_size, action_size, hidden_layer1, hidden_layer2, learning_rate)

# lists for the states, actions and rewards
states = []
actions = []
rewards = []
scores = []
episodes = []
last_100_avg = []
episode_len = []

for episode in range(EPISODES):
    done = False
    score = 0
    state = env.reset()
    state = np.reshape(state, [1, org_state_size])
    previous_state = np.zeros((1, org_state_size))

    while not done:

        # Create a set of state variables that includes state at frame [f] and [f-1]
        # This gives the model access to the rate of change of the velocity terms in the specific case.
        state_holder = np.zeros((1, state_size))
        state_holder[0, 0:8] = previous_state[0]
        state_holder[0, 8::] = state[0]

        state = state_holder

        # get an action based on the current state from the network.
        action = action_selection(model, state, action_size)

        # step the environment forward with the selected action.
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, org_state_size])

        # record the state, the action taken in that state and reward received for taking that action.
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        # pull the current state out of the combined state representation
        previous_state = state[:, -org_state_size::]

        # add the reward to the running score for the episode
        score += reward

        # set the current state to the state transitioned to by environment after the selected action was taken.
        state = next_state

        if done:

            # after the episode ends keep track of scores and episodes elapsed
            scores.append(score)
            episodes.append(episode)

            # start to compute the running average the past 100 scores after the record is greater than 200 entries
            if len(scores) > 200:

                # calculate the average of the last 100 episodes
                last_100 = np.mean(scores[-100::])
                print "Last 100 Episodes Running Average:  {0}".format(last_100)

                # if the mean over the last 100 episodes is greater than 200
                if last_100 > 200:

                    # read from a csv what the current high score is.
                    a = pd.read_csv('Champion_batch.csv')
                    max_score = a['100meanScore'][0]

                    # if the current running average bests the current high score save the model as the new Champion.
                    if last_100 > max_score:
                        model.save_weights("Current_Champion_2.h5")

                        # record the settings used for that model
                        stats = pd.DataFrame(
                            columns=['Episode', '100meanScore', 'lr', 'discount', 'h1', 'h2', 'seed'])
                        stats['Episode'] = [episode]
                        stats['100meanScore'] = [last_100]
                        stats['lr'] = [learning_rate]
                        stats['discount'] = [discount_factor]
                        stats['h1'] = [hidden_layer1]
                        stats['h2'] = [hidden_layer2]
                        stats['seed'] = ['random']

                        stats.to_csv('Champion_batch.csv', index=False)

                        print "Current Max:   {0}".format(last_100)

            # after 100 episodes start recording the last 100 running average.
            if episode > 100:
                last_100_avg.append(np.mean(scores[-100::]))
            else:
                last_100_avg.append(np.mean(scores))

            print("episode:", episode, "  score:", score)
            episode_len.append(len(states))

            state_array.append(states)
            action_array.append(actions)
            reward_array.append(rewards)

            if episode % batch_size == 0:

                for i in range(len(state_array)):
                    # train the models and reset the states actions rewards history
                    states, actions, rewards = train_model(model, state_array[i], action_array[i], reward_array[i],
                                                           discount_factor, state_size, action_size)
                state_array = []
                action_array = []
                reward_array = []

    # Every 1000 episodes write the score history to a csv file
    if episode % 1000 == 0:
        collected_data = pd.DataFrame(columns=["Episodes", "Scores", "Last_100_avg"])
        collected_data['Episodes'] = episodes
        collected_data['Scores'] = scores
        collected_data['Last_100_avg'] = last_100_avg
        collected_data.to_csv('score_history_hLayers' + str(hidden_layer2) + '_discount' +
                              str(discount_factor) + '_lr' + str(learning_rate) + '_batch' + str(batch_size) + '.csv',
                              index=False)

env.close()

