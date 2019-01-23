from keras.models import clone_model
from keras.optimizers import SGD

import numpy as np
from PIL import Image
import cv2

def clone_keras_model(model, learning_rate):
    
    res = clone_model(model)
    res.set_weights(model.get_weights())
    res.compile(loss='msle',
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

def export_image(state, seq):

    i = np.asarray(state)
    img = Image.fromarray(i, 'RGB')
    img.save('images/my_{}.png'.format(seq))

def normalize_image(state):
    img = Image.fromarray(np.asarray(state), 'RGB')
    ig = np.array(img.convert('L'))
    normalized = np.zeros((66, 200))
    normalized = cv2.normalize(ig, normalized, 0, 255, cv2.NORM_MINMAX)

    return normalized