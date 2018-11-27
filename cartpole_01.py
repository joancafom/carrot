from keras.models import Sequential
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
    
if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('CartPole-v0')
    print(env)
    agent = DQNAgent(4,2)
    episodes = 10000
    # Iterate the game
    for e in range(episodes):
        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, 4])
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(500):
            # turn this on if you want to render
            # env.render()
            # Decide action
            action = agent.act(state)
            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, episodes, time_t))
                break
        # train the agent with the experience of the episode
        agent.replay(32)