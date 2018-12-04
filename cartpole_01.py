from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import gym

# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):

        # State_size --> Número de entradas de la RN
        self.state_size = state_size

        # Action_size --> Número de salidas de la RN (posibles acciones)
        self.action_size = action_size

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
        # Neural Net for Deep-Q learning Model

        # Modelo de RN Secuencial (Feed-Forward)
        model = Sequential()
        
        # 1ra Capa de entrada, .....
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))

        #2da Capa intemedia, ...
        model.add(Dense(24, activation='relu'))

        #3ra Capa de salida, con dos nodos representando las acciones
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, done):
        # Lista de experiencias previas. Remember se usa para añadir
        # una experiencia a la memoria
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):

        # Actuamos aleatoriamente si no se supera la probabilidad
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Predecimos la recompensa de todas las acciones aplicadas sobre el estado dado
        act_values = self.model.predict(state)

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
    
if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('CartPole-v1')

    # Número de entradas (información del estado) y salidas (posibles acciones)
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]

    # Número de simulaciones a ejecutar
    episodes = 1000

    agent = DQNAgent(state_size,action_size)

    done = False
    batch_size = 32
    #agent.load("cartpole-dqn.h5")

    # Iterate the game
    for e in range(episodes):

        # reset state in the beginning of each game
        # Por cada episodio, empezamos de nuevo
        state = env.reset()

        #Conversión de la información del estado a matriz 1x4
        state = np.reshape(state, [1, state_size])

        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score

        # time_t -> Objetivo temporal a conseguir, por cada frame obtenemos un punto
        # El episodio se acaba cuando llegamos a ese objetivo o la pértiga se cae
        for time_t in range(500):

            # turn this on if you want to render

            # env.render()

            # Decide action
            #Decidir una acción basándose en el estado
            action = agent.act(state)

            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived

            # Obtener el siguiente estado usando la acción a realizar
            next_state, reward, done, _ = env.step(action)
            #Conversión a matriz de 1x4
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done else -10
            # Remember the previous state, action, reward, and done
            # Añadir a la memoria el estado, la acción tomada, la recompensa, el 
            # siguiente estado y si se finalizó el juego o no
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            #Tras realizar la acción nos encontramos en el próximo estado
            state = next_state

            # done becomes True when the game ends
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}, e:{:.2}"
                      .format(e, episodes, time_t, agent.epsilon))
                break
            # train the agent with the previous experiences (each frame)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        if e % 10 == 0:
            agent.save("cartpole-dqn.h5")
    print("Guardado")