from keras.models import clone_model
from keras.optimizers import SGD

import numpy as np
from PIL import Image
from collections import deque

def clone_keras_model(model, learning_rate):
    
    res = clone_model(model)
    res.set_weights(model.get_weights())
    res.compile(loss='mse',
                optimizer=SGD(lr=learning_rate))
    return res


def convert_action_to_gym(action):
    """Converts an action that was output by the NN
    and transforms it into a valid one to use in Gym"""

    #NN's Actions --> ['Izquierda', 'Centro', 'Derecha', 
    #                'Izq-Gas', 'Centro-Gas', 'Dcha-Gas']
    #res --> actions of the gym car [steer, gas, break]
   
    res = [0,0,0]
    
    if action == 0:
        #Izquierda
        res[0] = -0.5

    elif action == 1:
        #Centro
        pass

    elif action == 2:
        #Derecha
        res[0] = 0.5

    elif action == 3:
        #Izquierda + Gas
        res[0] = -0.5
        res[1] = 0.5

    elif action == 4:
        #Centro + Gas
        res[1] = 0.5

    elif action == 5:
        #Derecha + Gas
        res[0] = 0.5
        res[1] = 0.5
    
    return res


def convert_action_to_nn(gym_action):
    """Converts a gym_action to a valid one
    for the NN"""

    #NN's Actions --> ['Izquierda', 'Centro', 'Derecha', 
    #                'Izq-Gas', 'Centro-Gas', 'Dcha-Gas']
    #res --> actions of the gym car [steer, gas, break]
   
    # By default, we return Centro
    res = 1

    # Legacy support for brake...
    if gym_action[2] != 0.0:
        print('You should not use this action')
        return 6

    # We check if 'Gas' is enabled
    if gym_action[1] == 0.0:
        # Gas disabled
        if gym_action[0] == -1.0:
            # Left
            res = 0
        elif gym_action[0] == 1.0:
            # Rigth
            res = 2
    else:
        # Gas enabled
        if gym_action[0] == -1.0:
            # Left + Gas
            res = 3
        elif gym_action[0] == 0.0:
            # Centre + Gas
            res = 4
        else:
            # Right + Gas
            res = 5

    return res

def export_image(state, seq):

    denormalized_state = state * 255.0
    i = denormalized_state.astype(np.uint8)
    
    #Triple Grayscale Image
    imgGray = Image.fromarray(i, 'RGB')

    #Single Binarized Grayscale
    i = np.array(imgGray.convert('L'))
    i = binarize(i)
    imgL = Image.fromarray(i, 'L')

    imgGray.save('images/my_{}_Triple.png'.format(seq))
    imgL.save('images/my_{}_Binarized.png'.format(seq))

def binarize(image_array, threshold = 50):
    b_array = np.array(image_array)
    for i in range(len(b_array)):
        for j in range(len(b_array[0])):
            if b_array[i][j] > threshold:
                b_array[i][j] = 255
            else:
                b_array[i][j] = 0
    
    return b_array

# Image Preprocessing for the CNN
def preprocess_image(rgb_image):
    
    # 1) Grayscale
    res = np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])
    # 2) Crop the screen
    #res = res[29:95]
    # 3) Binarize
    res = binarize(res)
    # 4) Normalize
    res = res/255.0

    return res

# freecodecamp
def stack_frames(stacked_frames, state, is_new_episode, state_size_h=66, state_size_w=200, state_size_d=3, stack_size=3):

    #Preprocess frame
    frame = preprocess_image(state)

    if is_new_episode:
        # Clear out stacked_frames
        stacked_frames = deque([np.zeros((state_size_h,state_size_w), 
        dtype=np.uint8) for i in range(stack_size)], maxlen=state_size_d)

        # Because we're in a new episode, copy the same frame 3x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames into a single image
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)
    
    return stacked_state, stacked_frames

# Escritura de un pack

import os
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def prepare_dir():

    RECORD_DIRECTORY = os.path.join(BASE_DIR, 'records')

    if not os.path.exists(RECORD_DIRECTORY):
        os.makedirs(RECORD_DIRECTORY)

# Patrón: pack_{ddMMYYYYHHmmss}.carrots
def write_pack(state, action, reward, next_state, done, file_date):

    get_file_path = lambda x: os.path.join(BASE_DIR, 'records/pack_{}.carrots')

    with open(get_file_path(file_date), mode='a', encoding='utf-8') as pack:

        file_writer = csv.writer(pack, delimiter='|')
        file_writer.writerow(['State', 'Action', 'Reward', 'Next State', 'Done'])

        
    return None