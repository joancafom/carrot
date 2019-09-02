from collections import deque
import numpy as np

'''

    This class is based on the one provided by Branko Blagojevic
    in the article that can be found in:

    Esta clase está basada en la especificada por Branko Blagojevic
    que puede ser encontrada en el artículo:
    
    https://medium.com/ml-everything/learning-from-pixels-and-deep-q-networks-with-keras-20c5f3a78a0

'''

class ExperienceReplay:
    
    def __init__(self, buffer_size=10000):
        """ Data structure used to hold game experiences """
        
        # Buffer will contain [state, action, reward, next_state, done]
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
    
    def add(self, experience):
        """ Adds list of experiences to the buffer """

        # Deque keeps the last buffer_size experiencies
        self.buffer.extend(experience)
        
    def sample(self, size):
        """ Returns a sample of experiences from the buffer """

        # Instead of using np.random.choice (shuffles the entire array
        # before getting the sample) we create a list of random ints (ids)
        # we will use to get the experiences
        sample_idxs = np.random.randint(len(self.buffer), size=size)
        sample_output = [self.buffer[idx] for idx in sample_idxs]
        sample_output = np.reshape(sample_output,(size, -1))

        return sample_output