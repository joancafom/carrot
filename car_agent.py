from __future__ import division

from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, Lambda
import keras.backend as K

import numpy as np
import random

import os
from collections import deque


'''
Double Deep Q-Learning, Dueling Agent and Fixed Q-Targets
'''



class CarAgent:

    def __init__(self, load_models):

        # Definition of the different actions our agent can perform
        #   - Index: ID of the action. i.e: 0->'Izquierda'
        #   - Value: A human description of the action
        self.actions = ['Izquierda', 'Centro', 'Derecha', 'Centro-Gas']

        # Dimensions of a single image that will form a stack
        self.state_size_h = 134
        self.state_size_w = 200
        self.state_size_d = 3

        # Stack of images that will be fed to the network 
        self.stack_size = self.state_size_d
        self.stacked_frames = deque([np.zeros((self.state_size_h,self.state_size_w), dtype=np.uint8) for i in range(self.stack_size)], maxlen=self.state_size_d)

        # ----- Hyperparameters -----

        # Network
        self.final_conv_layer_size = 512
        self.kernel_initializer = 'glorot_normal'
        self.optimizer = 'adam'
        self.loss_function = 'mse'
        self.learning_rate = 0.0001

        # Rate/Percentage of update that is applied
        # when we transfer weights from one network
        # to another
        self.tau = 1

        
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
    
    # Google's Deep-Mind
    # Fixed Q-Targets
    # Double DQN Update
    def __transfer_network_weights__(self, source_nn, target_nn, tau):

        weights_to_transfer = (np.array(source_nn.get_weights()) * tau) + \
        (np.array(target_nn.get_weights()) * (1 - tau))

        target_nn.set_weights(weights_to_transfer)

    def update_target_network(self, tau=1):
        self.__transfer_network_weights__(self.main_qn, self.target_qn, tau)

