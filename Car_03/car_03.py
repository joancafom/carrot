import gym
import os
import numpy as np
from agent import DQNAgent
from aux import convert_action_to_gym

file_identifier = 'car_' + '03'
weights_file = os.path.join(os.path.dirname(__file__), file_identifier + '-dqn.h5')
    
def train():
    
    # initialize gym environment and the agent
    env = gym.make('CarRacing-v0')

    # Número de simulaciones a ejecutar
    episodes = 1000
    time_limit = 250

    agent = DQNAgent()

    done = False
    batch_size = 32

    #agent.load(weights_file)

    # Iterate the game
    for e in range(episodes):

        # reset state in the beginning of each game
        # Por cada episodio, empezamos de nuevo
        state = env.reset()
        agent.epsilon = 1.0

        # Reward acumulado
        cumulated_reward = 0

        # time_t -> Objetivo temporal a conseguir
        # El episodio se acaba cuando todas las tiles se visitan o se consume el tiempo (gym)
        for time_t in range(time_limit):
            
            #Conversión de la información de la imagen de estado a matriz 1x200x60x3 (RGB)
            state = np.reshape(state, [1, agent.state_size_h, agent.state_size_w,agent.state_size_d])

            # turn this on if you want to render
            env.render()

            #Decidir una acción basándose en el estado
            action_to_perform = agent.act(state)

            #Convertimos la salida de nuestra RN al formato que espera gym para actuar
            gym_action_to_perform = convert_action_to_gym(action_to_perform)

            # Obtener el siguiente estado usando la acción a realizar
            next_state, reward, done, _ = env.step(gym_action_to_perform)

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
                      .format(e, episodes, time_t, agent.epsilon, cumulated_reward, agent.actions_meaning[action_to_perform]))
                break
            else:
                # print the score and break out of the loop
                print("Non-episode: {}/{}, frame: {}, e:{:.2}, score: {}, action: {}"
                      .format(e, episodes, time_t, agent.epsilon, cumulated_reward, agent.actions_meaning[action_to_perform]))

            # train the agent with the previous experiences (each frame)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        #Save the weights of the NN each 5 episodes
        if e % 5 == 0:
            agent.save(weights_file)
            print("Guardado")

def play():
    
    # initialize gym environment and the agent
    env = gym.make('CarRacing-v0')

    # Número de simulaciones a ejecutar
    episodes = 1000
    time_limit = 500

    agent = DQNAgent()
    agent.load(weights_file)

    # Iterate the game
    for e in range(episodes):

        # reset state in the beginning of each game
        # Por cada episodio, empezamos de nuevo
        state = env.reset()

        #La proporción de acciones será la establecida comom
        #mínima
        agent.epsilon = agent.epsilon_min

        # Reward acumulado
        cumulated_reward = 0

        # time_t -> Objetivo temporal a conseguir
        # El episodio se acaba cuando todas las tiles se visitan o se consume el tiempo (gym)
        for time_t in range(time_limit):
            
            #Conversión de la información de la imagen de estado a matriz 1x200x60x3 (RGB)
            state = np.reshape(state, [1, agent.state_size_h, agent.state_size_w,agent.state_size_d])

            # turn this on if you want to render
            env.render()

            #Decidir una acción basándose en el estado
            action_to_perform = agent.act(state)

            #Convertimos la salida de nuestra RN al formato que espera gym para actuar
            gym_action_to_perform = convert_action_to_gym(action_to_perform)

            # Obtener el siguiente estado usando la acción a realizar
            next_state, reward, done, _ = env.step(gym_action_to_perform)

            #Conversión a matriz de 1x200x60x3 (el primer número indica el batch/sample al que pertenece la imagen)
            next_state = np.reshape(next_state, [1, agent.state_size_h, agent.state_size_w, agent.state_size_d])

            # make next_state the new current state for the next frame.
            #Tras realizar la acción nos encontramos en el próximo estado
            state = next_state
            cumulated_reward += reward

            # done becomes True when the game ends
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, frame: {}, e:{:.2}, score: {}, action: {}"
                      .format(e, episodes, time_t, agent.epsilon, cumulated_reward, agent.actions_meaning[action_to_perform]))
                break
            else:
                # print the score and break out of the loop
                print("Non-episode: {}/{}, frame: {}, e:{:.2}, score: {}, action: {}"
                      .format(e, episodes, time_t, agent.epsilon, cumulated_reward, agent.actions_meaning[action_to_perform]))

def print_model():

    agent = DQNAgent()
    agent.load(weights_file)
    agent.print_model_graph('car_model.png')

        
if __name__ == "__main__":
    play()
