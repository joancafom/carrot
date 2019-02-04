from __future__ import division

from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, Lambda
import keras.backend as K

import numpy as np
import random

import os
from collections import deque
from experience_replay import ExperienceReplay


'''
Double Deep Q-Learning, Dueling Agent and Fixed Q-Targets
'''



class CarAgent:

    def __init__(self, load_models):

        # Definition of the different actions our agent can perform
        #   - Index: ID of the action. i.e: 0->'Izquierda'
        #   - Value: A human description of the action
        self.actions = ['Izquierda', 'Centro', 'Derecha', 'Centro-Gas', 'Freno']

        # Dimensions of a single image that will form a stack
        self.state_size_h = 134
        self.state_size_w = 200
        self.state_size_d = 3

        # Stack of images that will be fed to the network 
        self.stack_size = self.state_size_d
        self.stacked_frames = deque([np.zeros((self.state_size_h,self.state_size_w), dtype=np.uint8) for i in range(self.stack_size)], maxlen=self.state_size_d)

        self.experience_buffer = ExperienceReplay()

        # ----- Hyperparameters -----

        # Network
        # Size of the final convolution layer before 
        # splitting into Advantage and Value streams
        self.final_conv_layer_size = 512
        self.kernel_initializer = 'glorot_normal'
        self.optimizer = 'adam'
        self.loss_function = 'mse'
        self.learning_rate = 0.0001

        # Rate/Percentage of update that is applied
        # when we transfer weights from one network
        # to another
        self.tau = 1

        # Random ratio
        self.epsilon = 0.6
        self.epsilon_min = 0.1
        # Discount rate: Devaluation of the 
        # future actions reward
        self.gamma = 0.99

        
        # ----- Networks -----

        # Main Q-Network
        self.main_qn = self.__build_model__()
        # Target Q-Network
        self.target_qn = self.__build_model__()

        # Make both networks equal
        self.update_target_network(tau=self.tau)

        # ----- Saving -----

        # Where to save our models
        self.weights_folder = './models'
        self.main_weights_file = self.weights_folder + "/main_weights.h5"
        self.target_weights_file = self.weights_folder + "/target_weights.h5"

        # If the path does not exists, we create it
        if not os.path.exists(self.weights_folder):
            os.makedirs(self.weights_folder)

        # We load the saved models if required
        if load_models:
            if os.path.exists(self.main_weights_file):
                print("Loading main weights...")
                self.main_qn.load_weights(self.main_weights_file)
            if os.path.exists(self.target_weights_file):
                print("Loading target weights...")
                self.target_qn.load_weights(self.target_weights_file)

    def __build_model__(self):

         # The input of the NN will be the stacked frames dimensions
         # That is:
         #          -   Height: Image/State's height
         #          -   Width: Image/State's width
         #          -   Depth: Number of stacked frames (3 by default)
        inputs = Input(shape=(self.state_size_h, self.state_size_w, self.stack_size), name="main_input")

        # There will be four layers of convolutions performed on the stack of images input
        model = Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation="relu",
            padding="valid", kernel_initializer=self.kernel_initializer, name="conv1")(inputs)

        model = Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation="relu",
            padding="valid", kernel_initializer=self.kernel_initializer, name="conv2")(model)
        
        model = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation="relu",
            padding="valid", kernel_initializer=self.kernel_initializer, name="conv3")(model)
        
        model = Conv2D(filters=self.final_conv_layer_size, kernel_size=(7,7), strides=(1,1), activation="relu",
            padding="valid", kernel_initializer=self.kernel_initializer, name="conv4")(model)
        
        # Dueling DQN

        # We then separate the final convolution layer into an advantage and value
        # stream. The value function is how well off you are in a given state. Every 
        # state has its associated value (i.e. being off the road). From that 
        # point, the advantage represents how much better off you end up after performing
        # one action in that state. Q is the value function of a state after a given action.
        # Advantage(state, action) = Q(state, action) - Value(state)
        # Q values is now easier to compute
        # We give each stream half of the final Conv2D output
        #   Advantage Stream Compute (AC): 0->(final_conv_layer_size // 2)
        #   Value Stream Compute (VC): (final_conv_layer_size // 2) -> final_conv_layer_size
        stream_AC = Lambda(lambda layer: layer[:,:,:,:self.final_conv_layer_size // 2], name="advantage")(model)
        stream_VC = Lambda(lambda layer: layer[:,:,:,self.final_conv_layer_size // 2:], name="value")(model)
        
        # We then flatten the advantage and value functions: We transform
        # the depth to be just 1
        stream_AC = Flatten(name="advantage_flatten")(stream_AC)
        stream_VC = Flatten(name="value_flatten")(stream_VC)
        
        # We define weights for our advantage and value layers. We will train these
        # layers so the matmul will match the expected value and advantage from play
        # Remember that the advantage references each action
        # The value is just how well we are in a single state
        advantage_layer = Dense(len(self.actions),name="advantage_final")(stream_AC)
        value_layer = Dense(1, name="value_final")(stream_VC)

        # To get the Q output, we need to add the value to the advantage.
        # But adding them directly is not correct! (Given Q we're unable to
        # find A(s,a) and V(s))
        #   Q(s,a) != V(s) + A(s, a)
        # We can solve this by substracting the mean of the Advante to A(s,a)
        model = Lambda(lambda val_adv: val_adv[0] + 
            (val_adv[1] - K.mean(val_adv[1], axis=1, keepdims=True)), 
            name="final_out")([value_layer, advantage_layer])

        model = Model(inputs, model)
        model.compile(self.optimizer, self.loss_function)
        model.optimizer.lr = self.learning_rate

        return model
    
    def train(self, batch_size):
        
        # Train batch is [[state,action,reward,next_state,done],...]
        # That is an array of 64 x Experience
        # An Experience = [state, action, reward, next_state, done]
        # = 1 x 5
        # So basically, a train batch has dims 64 x 1 x 5
        #
        # We also have to consider:
        #   state = 134 x 200 x 3
        #   action = 1
        #   reward = 1
        #   next_state = 134 x 200 x 3
        #   done = 1
        #
        train_batch = self.experience_buffer.sample(batch_size)

        # We can transpose the array in order to have access
        # to the individual components
        # dim [64 x 1 x 5]^T = dim 5 x 1 x 64
        # Now we have access to the different packs 
        train_state, train_action, train_reward, \
            train_next_state, train_done = train_batch.T

        # Convert the action array into an array of ints so they can be used for indexing
        train_action = train_action.astype(np.int)

        # Stack the train_state and train_next_state for learning
        # reshape it to have 64 x (134 x 200 x 3) (stack it in vertical)
        train_state = np.vstack(train_state)
        train_next_state = np.vstack(train_next_state)

        # Our predictions (actions to take) from the main Q network
        # Regular Q(s, *; w) prediction
        regular_q = self.main_qn.predict(train_state)
        
        # The main network chooses the next action with the highest Q
        # Q(s', *; w)
        main_next_q = self.main_qn.predict(train_next_state)
        # next_action = a' = argmax Q(s', *; w)
        next_action = np.argmax(main_next_q, axis=1)
        next_action = next_action.astype(np.int)
        
        # Tells whether our game is over or not
        # If our game has ended, we do not compute the future discounted
        # reward
        train_gameover = train_done == 0

        # The target network is now used to estimate the Q values
        # of taking that action in the next state
        # target_next_state = Q(s', *; w-)
        target_next_q = self.target_qn.predict(train_next_state)
        next_state_values = target_next_q[range(batch_size), next_action]

        # Reward from the action chosen in the train batch
        actual_reward = train_reward + (self.gamma * next_state_values * train_gameover)
        
        # Update the prediction the main nn would ouput with 
        # the new values to perform a gradient descent step
        # regular_q(a) = y_j
        regular_q[range(batch_size), train_action] = actual_reward

        # Train the main model
        loss = self.main_qn.train_on_batch(train_state, regular_q)
        
        return loss

    # Google's Deep-Mind
    # Fixed Q-Targets
    def __transfer_network_weights__(self, source_nn, target_nn, tau):

        weights_to_transfer = (np.array(source_nn.get_weights()) * tau) + \
        (np.array(target_nn.get_weights()) * (1 - tau))

        target_nn.set_weights(weights_to_transfer)

    def update_target_network(self, tau=1):
        self.__transfer_network_weights__(self.main_qn, self.target_qn, tau)
    
    def get_action(self, state, is_random=False):
        
        if np.random.rand() < self.epsilon or is_random:
            # Act randomly if the threshold is not surpassed
            # or if we want it to be random.
            return np.random.randint(len(self.actions))

        else:
            # Predict the action with the highest Q using the
            # nn
            return np.argmax(self.main_qn.predict(np.array([state])))

    def save_models(self):
        print('Saving models\' weights...')
        try:
            self.main_qn.save_weights(self.main_weights_file)
            self.target_qn.save_weights(self.target_weights_file)

        except Exception as e:
            print('There was an error while trying to save the weights')
            print(str(e))
