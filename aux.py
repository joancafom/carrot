from keras.models import clone_model
from keras.optimizers import SGD

import numpy as np
from PIL import Image

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

    i = np.array(state)
    imgRgb = Image.fromarray(i, 'RGB')
    i = np.array(imgRgb.convert('L'))
    i = binarize(i)
    imgL = Image.fromarray(i, 'L')

    imgRgb.save('images/my_{}_RGB.png'.format(seq))
    imgL.save('images/my_{}_L.png'.format(seq))

def binarize(image_array, threshold = 50):
    b_array = np.array(image_array)
    for i in range(len(b_array)):
        for j in range(len(b_array[0])):
            if b_array[i][j] > threshold:
                b_array[i][j] = 255
            else:
                b_array[i][j] = 0
    
    return b_array