import numpy as np
from PIL import Image
from collections import deque

'''
    Module with several functions of different functionalities
    that are helpful in the training/AI part of our project.
'''

# ----- Actions conversion -----

def convert_action_to_gym(action):
    """Converts an action that was output by the NN
    and transforms it into a valid one to use in Gym"""

    #NN's Actions --> ['Izquierda', 'Centro', 'Derecha', 'Centro-Gas', 'Freno']
    #res --> Actions of the gym car [steer, gas, break]
   
    res = [0.0, 0.0, 0.0]
    
    if action == 0:
        # Izquierda
        res[0] = -1.0

    elif action == 1:
        # Centro
        pass

    elif action == 2:
        # Derecha
        res[0] = +1.0

    elif action == 3:
        # Centro-Gas
        res[1] = +1.0

    elif action == 4:
        # Freno
        res[2] = +0.8
    
    return res

def convert_action_to_nn(gym_action):
    """Converts a gym_action to a valid one
    for the NN"""

    #NN's Actions --> ['Izquierda', 'Centro', 'Derecha', 'Centro-Gas', 'Freno']
    #res --> actions of the gym car [steer, gas, break]
   
    # By default, we return 'Centro'
    res = 1

    # Freno Detection
    if gym_action[2] != 0.0:
        return 4

    # We check if 'Gas' is enabled
    if gym_action[1] == 0.0:
        # Gas disabled
        if gym_action[0] == -1.0:
            # Izquierda
            res = 0
        elif gym_action[0] == 1.0:
            # Derecha
            res = 2
    else:
        # Gas enabled
        if gym_action[0] != 0.0:
            # Izq + Gas or Dcha + Gas
            # This should not be used
            print('This action is not supported... \n')
            print('It will be replaced with Centro-Gas')

        res = 3

    return res



# ----- Image processing -----

def binarize(image_array, threshold = 50):
    '''
    Converts a grayscale image into an image that only
    contains either black (0) or white (255)

    Returns: a numpy array representing the binarized image
    '''

    b_array = np.array(image_array)
    for i in range(len(b_array)):
        for j in range(len(b_array[0])):
            if b_array[i][j] > threshold:
                b_array[i][j] = 255
            else:
                b_array[i][j] = 0
    
    return b_array

def export_image(state, seq):
    '''
    Saves the given `state` to two images. One contains
    one stacked frame per RGB channel (3 in total) and the 
    other is just its binarization.
    '''

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

def preprocess_image(rgb_image):
    '''
    Prepares an RGB for the NN.
    Grayscale + Binarization + Normalization
    '''
    # 1) Grayscale
    res = np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])
    # 2) Crop the screen
    #res = res[29:95]
    # 3) Binarize
    res = binarize(res)
    # 4) Normalize
    res = res/255.0

    return res

# Based on the code from freecodecamp
def stack_frames(stacked_frames, state, is_new_episode, state_size_h=66, state_size_w=200, state_size_d=3, stack_size=3):
    '''
    Processes the state and uses it to produce a stack with the last `stack_size` images in memory
    
    Returns: 
        - the stacked frames, ready to be processed by the nn
        - the frames used to produce the stacked frames
    '''

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