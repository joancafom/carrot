from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam, SGD
from collections import deque
import numpy as np
import random
import gym
from keras import backend as K
import cv2
from PIL import Image

from gym.envs.box2d.car_racing import CarRacing

actions_meaning = ['Izquierda', 'Centro', 'Derecha', 'Izq-Gas', 'Centro-Gas', 'Dcha-Gas']

# Deep Q-learning Agent
class DQNAgent:
    def __init__(self):

        # State_size_h --> Altura de la imagen
        self.state_size_h = 66

        # State_size_w --> Anchura de la imagen
        self.state_size_w = 200

        # State_size_d --> Profundidad de la imagen
        self.state_size_d = 3

        # Action_size --> Número de salidas de la RN (posibles acciones)
        # Descritas en actions_meaning
        self.action_size = len(actions_meaning)

        # Memoria rápida para almacenar estados anteriores 
        self.memory = deque(maxlen=2000)

        # Depreciación del reward de acciones lejanas
        self.gamma = 0.95    # discount rate

        # Ratio de acciones que tomamos aleatoriamente
        self.epsilon = 1.0  # exploration rate

        # Mínimo de acciones aleatorias
        self.epsilon_min = 0.01

        # Coeficiente de depreciación de la aleatoriedad 
        self.epsilon_decay = 0.995

        # Cuánto aprende una RN en cada iteración
        self.learning_rate = 0.001

        # La RN
        self.model = self._build_model()


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
            return random.randrange(0, len(actions_meaning))

        # Predecimos la recompensa de todas las acciones aplicadas sobre el estado dado
        act_values = self.model.predict(state)
        print("Valores: " + str(act_values))
        # Escogemos realizar la acción con más recompensa
        return np.argmax(act_values[0])  # returns action


    def replay(self, batch_size):

        # Muestra aleatoria de la memoria de tamaño batch_size
        minibatch = random.sample(self.memory, batch_size)

        #Por cada recuerdo en la muestra
        for state, action, reward, next_state, done in minibatch:
            
            # target --> la recompensa máxima posible para un estado
            # if done: el objetivo es la recompensa 
            target = reward

            if not done:
                
                # El objetivo es la recompensa actual más el máximo entre las predicciones del siguiente estado (depreciadas)
                target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            
            # Predicción del objetivo (recompensa que se obtiene por cada acción)
            target_f = self.model.predict(state)

            #
            target_f[0][action] = target

            # Aplicamos la
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
    
def convert_action_to_gym(action):
   
    #acciones de la RN = [izq, centro, dcha]
    #res --> actions of the gym car [steer, gas, break]
    #actions_meaning = ['Izquierda', 'Centro', 'Derecha', 'Izq-Gas', 'Centro-Gas', 'Dcha-Gas']
   
    res = [0,0,0]
    
    if action == 0:
        res[0] = -0.5
    elif action == 1:
        pass
    elif action == 2:
        res[0] = 0.5
    elif action == 3:
        res[0] = -0.5
        res[1] = 0.5
    elif action == 4:
        res[0] = 0
        res[1] = 0.5
    elif action == 5:
        res[0] = 0.5
        res[1] = 0.5
    
    return res


if __name__ == "__main__":
    
    # initialize gym environment and the agent
    env = gym.make('CarRacing-v0')

    # Número de simulaciones a ejecutar
    episodes = 10

    agent = DQNAgent()

    done = False
    batch_size = 32

    agent.load("cartpole-dqn.h5")

    # Iterate the game
    for e in range(episodes):

        # reset state in the beginning of each game
        # Por cada episodio, empezamos de nuevo
        state = env.reset()
        agent.epsilon = 0.5

        # Reward acumulado
        cumulated_reward = 0

        # time_t -> Objetivo temporal a conseguir, por cada frame obtenemos un punto
        # El episodio se acaba cuando todas las tiles se visitan o se consume el tiempo (gym)
        for time_t in range(250):
            
            #Conversión de la información de la imagen de estado a matriz 1x200x60x3 (RGB)
            if time_t == 0:
                state = state[29:95]
            state = np.reshape(state, [1, agent.state_size_h, agent.state_size_w,agent.state_size_d])

            # turn this on if you want to render

            env.render()

            #Decidir una acción basándose en el estado
            action_to_perform = agent.act(state)

            #Convertimos la salida de nuestra RN al formato que espera gym para actuar
            gym_action_to_perform = convert_action_to_gym(action_to_perform)

            # Obtener el siguiente estado usando la acción a realizar
            next_state, reward, done, _ = env.step(gym_action_to_perform)

            next_state = next_state[29:95]

            if time_t == 50:
                i = np.asarray(next_state)
                img = Image.fromarray(i, 'RGB')
                img.save('my.png')


            #Conversión a matriz de 1x200x60x3 (el primer número indica el batch/sample al que pertenece la imagen)
            
            next_state = np.reshape(next_state, [1, agent.state_size_h, agent.state_size_w, agent.state_size_d])

            # Añadir a la memoria el estado, la acción tomada, la recompensa, el 
            # siguiente estado y si se finalizó el juego o no
            agent.remember(state, action_to_perform, reward, next_state, done)

            # make next_state the new current state for the next frame.
            #Tras realizar la acción nos encontramos en el próximo estado
            state = next_state
            cumulated_reward += reward

            # done becomes True when the game ends
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, frame: {}, e:{:.2}, score: {}, action: {}"
                      .format(e, episodes, time_t, agent.epsilon, cumulated_reward, actions_meaning[action_to_perform]))
                break
            else:
                # print the score and break out of the loop
                print("Non-episode: {}/{}, frame: {}, e:{:.2}, score: {}, action: {}"
                      .format(e, episodes, time_t, agent.epsilon, cumulated_reward, actions_meaning[action_to_perform]))

            
            # train the agent with the previous experiences (each frame)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # if time_t == 50:

            #     state = np.reshape(state, [agent.state_size_h, agent.state_size_w,agent.state_size_d])
            #     i = np.asarray(state)
            #     img = Image.fromarray(i, 'RGB')
            #     img.save('my2.png')

        
        #if e % 10 == 0:
        agent.save("cartpole-dqn.h5")
        print("Guardado")

    def get_output_layer(model, layer_name):
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        layer = layer_dict[layer_name]
        return layer

    def global_average_pooling(x):
        return K.mean(x, axis = (2, 3))
    
    def global_average_pooling_shape(input_shape):
        return input_shape[0:2]
	     
    def visualize_class_activation_map(img_path, output_path):
        agent = DQNAgent()
        agent.load("cartpole-dqn.h5")
        model = agent.model()

        original_img = cv2.imread(img_path, 1)
        width, height, _ = original_img.shape

        #Reshape to the network input shape (3, w, h).
        img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])
        
        #Get the 512 input weights to the softmax.
        class_weights = model.layers[-1].get_weights()[0]
        final_conv_layer = get_output_layer(model, "conv5_3")
        get_output = K.function([model.layers[0].input], \
                    [final_conv_layer.output, 
        model.layers[-1].output])
        [conv_outputs, predictions] = get_output([img])
        conv_outputs = conv_outputs[0, :, :, :]

        #Create the class activation map.
        cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1:3])
        for i, w in enumerate(class_weights[:, 1]):
            cam += w * conv_outputs[i, :, :]
        print("predictions")
        print(predictions)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0
        img = heatmap*0.5 + original_img
        cv2.imwrite(output_path, img)