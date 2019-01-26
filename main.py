import gym
import os
import numpy as np
from car_agent import DQNAgent
from aux import convert_action_to_gym, export_image, stack_frames


# ----- Identificadores para los archivos ------ 

file_identifier = 'car_' + 'main'
weights_file = os.path.join(os.path.dirname(__file__), file_identifier + '-dqn.h5')


# ----- Método usado para entrenar ------ 

def train(load_weights=True, frame_to_export=None, learn=True):
    
    # initialize gym environment and the agent
    env = gym.make('CarRacing-v0')

    # Número de simulaciones a ejecutar
    episodes = 100
    time_limit = 250

    agent = DQNAgent()

    done = False
    batch_size = 32

    # Si queremos cargar los pesos y el fichero existe
    if load_weights and os.path.exists(weights_file):
        agent.load(weights_file)

    # Iterate the game
    for e in range(episodes):

        # reset state in the beginning of each game
        # Por cada episodio, empezamos de nuevo
        state = env.reset()
        agent.epsilon = 0.7

        # Reward acumulado
        cumulated_reward = 0

        # Unimos los 3 últimos frames
        stacked_state, stacked_frames = stack_frames(agent.stacked_frames, state, True)
        agent.stacked_frames = stacked_frames

        # time_t -> Objetivo temporal a conseguir
        # El episodio se acaba cuando todas las tiles se visitan o se consume el tiempo (gym)
        for time_t in range(time_limit):
            
            # Conversión de la información de la imagen de estado a una válida para la red (1 dimensión más)
            stacked_state_nn = np.reshape(stacked_state, [1, agent.state_size_h, agent.state_size_w,agent.state_size_d])

            # turn this on if you want to render
            env.render()

            #Decidir una acción basándose en el estado
            action_to_perform = agent.act(stacked_state_nn)

            #Convertimos la salida de nuestra RN al formato que espera gym para actuar
            gym_action_to_perform = convert_action_to_gym(action_to_perform)

            # Obtener el siguiente estado usando la acción a realizar
            next_raw_state, reward, done, _ = env.step(gym_action_to_perform)

            # Obtenemos el grupo de las 3 imágenes (stack)
            next_stacked_state, next_stacked_frames = stack_frames(agent.stacked_frames, next_raw_state, False)
            agent.stacked_frames = next_stacked_frames

            if time_t == frame_to_export:
                export_image(next_stacked_state, time_t)

            #Conversión a matriz de 1x200x66x3 (el primer número indica el batch/sample al que pertenece la imagen)
            next_stacked_state_nn = np.reshape(next_stacked_state, [1, agent.state_size_h, agent.state_size_w, agent.state_size_d])

            # Añadir a la memoria el estado, la acción tomada, la recompensa, el 
            # siguiente estado y si se finalizó el juego o no
            if learn:
                agent.remember(stacked_state_nn, action_to_perform, reward, next_stacked_state_nn, done)

            # make next_state the new current state for the next frame.
            #Tras realizar la acción nos encontramos en el próximo estado
            stacked_state = next_stacked_state
            cumulated_reward += reward

            # done becomes True when the game ends
            if done:
                # print the score and break out of the loop
                print("Fin: {}/{}, frame: {}, e:{:.2}, score: {}, action: {}"
                      .format(e, episodes, time_t, agent.epsilon, cumulated_reward, agent.actions_meaning[action_to_perform]))
                break
            else:
                # print the score and break out of the loop
                print("Episode: {}/{}, frame: {}, e:{:.2}, score: {}, action: {} \n"
                      .format(e, episodes, time_t, agent.epsilon, cumulated_reward, agent.actions_meaning[action_to_perform]))

            # train the agent with the previous experiences (each frame)
            if learn and len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        #Save the weights of the NN each 5 episodes
        if learn and e % 5 == 0:
            agent.save(weights_file)
            print("Guardado")
    

def play():
    train(learn=False)


if __name__ == "__main__":
    res = input('¿Quieres entrenar al coche? Y/N \n')
    
    if 'y' == res or 'Y' == res:
        train(frame_to_export=25)
    else:
        play()