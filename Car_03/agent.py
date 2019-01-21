from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from collections import deque
import numpy as np
import random
import gym

from aux import clone_keras_model

#Useful to modify in-game characteristics, like
#printing stats on the screen
#from gym.envs.box2d.car_racing import CarRacing

# Deep Q-learning Agent
class DQNAgent:

    def __init__(self):

        #Definition of the different actions our agent can perform
        self.actions_meaning = ['Izquierda', 'Centro', 'Derecha', 'Izq-Gas', 'Centro-Gas', 'Dcha-Gas', 'Freno']

        # State_size_h --> Altura de la imagen
        self.state_size_h = 66

        # State_size_w --> Anchura de la imagen
        self.state_size_w = 200

        # State_size_d --> Profundidad de la imagen
        self.state_size_d = 3

        # Action_size --> Número de salidas de la RN (posibles acciones)
        # Descritas en actions_meaning
        self.action_size = len(self.actions_meaning)

        # Memoria rápida para almacenar estados anteriores 
        self.memory = deque(maxlen=2000)

        # Depreciación del reward de acciones lejanas
        self.gamma = 0.95    # discount rate

        # Ratio de acciones que tomamos aleatoriamente
        self.epsilon = 1.0  # exploration rate

        # Mínimo de acciones aleatorias
        self.epsilon_min = 0.1

        # Coeficiente de depreciación de la aleatoriedad 
        self.epsilon_decay = 0.995

        # Cuánto aprende una RN en cada iteración
        self.learning_rate = 0.001

        #Delayed-Copy interval
        self.dcopy_interval = 4

        # La RN
        self.model = self._build_model()

        self.delayed_model = clone_keras_model(self.model, self.learning_rate)


    def _build_model(self):
        
        # Neural Net for Deep-Q learning Model based on PilotNet Architecture by NVidia

        # Modelo de RN Secuencial (Feed-Forward)
        model = Sequential()
        
        # 1ra Capa, Entrada de imágenes normalizadas
        model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu', input_shape=(self.state_size_h, self.state_size_w, self.state_size_d)))
        model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
        model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        #Investigar sobre marcha atrás
        #Acciones de la RN. Descritas en actions_meaning
        model.add(Dense(self.action_size, activation='relu'))

        #Función de pérdida: MSE (Mean Squared Error)
        #Investigar optimizer
        model.compile(loss='msle',
                      optimizer=SGD(lr=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, done):
        # Lista de experiencias previas. Remember se usa para añadir
        # una experiencia a la memoria
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):

        # Actuamos aleatoriamente si no se supera la probabilidad
        if np.random.rand() <= self.epsilon:
            return random.randrange(0, len(self.actions_meaning))

        # Predecimos la recompensa de todas las acciones aplicadas sobre el estado dado
        act_values = self.model.predict(state)
        print("Valores: " + str(act_values))
        # Escogemos realizar la acción con más recompensa
        return np.argmax(act_values[0])  # returns action


    def replay(self, batch_size):

        # Muestra aleatoria de la memoria de tamaño batch_size
        minibatch = random.sample(self.memory, batch_size)

        steps = 0

        #Por cada recuerdo en la muestra
        for state, action, reward, next_state, done in minibatch:
            
            steps += 1

            # target --> la recompensa máxima posible para un estado
            # if done: el objetivo es la recompensa 
            target = reward

            if not done:
                
                # El objetivo es la recompensa actual más el máximo entre las predicciones del siguiente estado (depreciadas)
                target = reward + self.gamma * \
                       np.amax(self.delayed_model.predict(next_state)[0])
            
            # Predicción del objetivo (recompensa que se obtiene por cada acción)
            target_f = self.model.predict(state)

            #
            target_f[0][action] = target

            # Aplicamos la
            self.model.fit(state, target_f, epochs=1, verbose=0)

            if steps >= self.dcopy_interval:
                steps = 0
                self.delayed_model.set_weights(self.model.get_weights())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
    
    def print_model_graph(self, file_name):
        plot_model(self.model, to_file=file_name)