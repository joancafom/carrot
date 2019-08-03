import gym
import keras.backend as K

import numpy as np
import matplotlib.pyplot as plt
import os

from aux import convert_action_to_gym, export_image, stack_frames, live_step
from record_gameplay import RECORD_MAIN_PATH, BASE_DIR
from open_gameplays import open_episode

from car_agent import CarAgent
from experience_replay import ExperienceReplay

import threading
import sys
from tcpServer import tcp_server, car_env

#from gym.envs.box2d import CarRacing

# Initialize the CarRacing environment
env = gym.make('CarRacing-v0')

# ----- Statistics section ----- 

# Tracks rewards per episode
rewards = []
# How often the statistics are printed
print_every = 10
# How often we dump the agent's weights
save_every = 5
# Tracks training losses
losses = [0]



# ----- Train & Play sections ----- 

def train(car, batch_size, num_epochs, update_freq, annealing_steps, 
        max_num_episodes, pre_train_episodes, max_num_step, goal):

    # Open a new thread with the TCP Server to obtain the current state image
    try:

        t = threading.Thread(target=tcp_server)
        t.start()
    
    except Exception as e:
        print("\n\n FATAL: Could not start TCP Server. Exiting \n\n")
        print(e)
        sys.exit()
    

    # In the first episode we need some time to start the camera
    # app. We stop the execution until we open the app and the tcp server
    # gets the first image. Then we can press any key to continue the
    # execution
    input(" Waiting for the client to connect. Press any key to continue...")

    # We'll begin by acting complete randomly. As we gain experience and improve,
    # we will begin reducing the probability of acting randomly, and instead
    # take the actions that our Q network suggests
    prob_random_drop = (car.epsilon - car.epsilon_min) / annealing_steps

    num_episode = 0
    while num_episode < max_num_episodes:

        # Create an experience replay buffer for the current episode
        episode_buffer = ExperienceReplay(buffer_size=max_num_step)

        # Take an image of the road
        state = car_env.get_state() # New code

        # Process the state as a stack of three images
        stacked_state, stacked_frames = stack_frames(car.stacked_frames, state, True)
        # Save the last frames used to produce the stack 
        # (they will be used to create the next one)
        car.stacked_frames = stacked_frames
        state = stacked_state

        # Whether the Game is complete or not
        done = False
        # Total reward obtained within the episode
        sum_rewards = 0
        # Current step of the episode
        cur_step = 0

        while cur_step < max_num_step and not done:

            cur_step += 1

            # Get the action to perform for the state
            action = car.get_action(state, is_random=(num_episode < pre_train_episodes))

            # Perform the action and retrieve the next state, reward and done
            next_state, reward, done = live_step(action)
            print("NS: {}, reward: {}, done: {}".format(next_state, reward, done))

            # Process the state as a stack of three images
            next_stacked_state, next_stacked_frames = stack_frames(car.stacked_frames, next_state, False)
            car.stacked_frames = next_stacked_frames
            next_state = next_stacked_state

            if cur_step == 50:
                export_image(next_stacked_state, cur_step)

            # Set up the episode to be stored in the episode buffer
            episode = np.array([[state],action,reward,[next_state],done])
            episode = episode.reshape(1,-1)

            # Store the experience in the episode buffer
            episode_buffer.add(episode)

            # Update the running rewards
            sum_rewards += reward

            # Update the state
            state = next_state

        # Once the episode's finished, we proceed to train the network
        if num_episode > pre_train_episodes:

            if car.epsilon > car.epsilon_min:
                # Drop the probability of a random action
                car.epsilon -= prob_random_drop

            if num_episode % update_freq == 0:
                for _ in range(num_epochs):
                    loss = car.train(batch_size)
                    losses.append(loss)

                # Update the target model with values from the main model
                car.update_target_network()

                if num_episode % save_every == 0:
                    # Save the model
                    car.save_models()

        # Increment the episode counter
        num_episode += 1

        # Dump the episode buffer to the main one
        car.experience_buffer.add(episode_buffer.buffer)
        rewards.append(sum_rewards)

        # Print the statistics
        if num_episode % print_every == 0:

            mean_loss = np.mean(losses[-(print_every * num_epochs):])

            print("Num episode: {} Mean reward: {:0.4f} Prob random: {:0.4f}, Loss: {:0.04f}".format(
                num_episode, np.mean(rewards[-print_every:]), car.epsilon, mean_loss))
            if np.mean(rewards[-print_every:]) >= goal:
                print("Training complete!")
                break

    car.save_models()


def train_s(car, batch_size, num_epochs, update_freq, verbose=False):

    records_path = os.path.join(BASE_DIR, RECORD_MAIN_PATH)
    
    num_episode = 0
    for direc in os.listdir(records_path):

        # Current step of the episode
        cur_step = 1

        # Create an experience replay buffer for the current episode
        # No recorded episode has more than 2000 states
        episode_buffer = ExperienceReplay(buffer_size=2000)
        
        print('\n--- Episode {} - {}'.format(num_episode, direc))

        episode_path = os.path.join(records_path, direc)
        for e_state, e_next_state, e_action, e_reward, e_done in open_episode(episode_path):

            if cur_step == 1:
                state = e_state

                # Process the state as a stack of three images
                stacked_state, stacked_frames = stack_frames(car.stacked_frames, state, True)

                # Save the last frames used to produce the stack 
                # (they will be used to create the next one)
                car.stacked_frames = stacked_frames
                state = stacked_state

                # Whether the Game is complete or not
                done = False
                # Total reward obtained within the episode
                sum_rewards = 0
                
            cur_step += 1

            action = e_action
            next_state, reward, done = e_next_state, e_reward, e_done

            if verbose:
                # Show a video with the processed frames, actions and rewards
                plt.imshow(state)
                plt.title('{} # {}'.format(car.actions[action], reward))
                plt.suptitle('R: {}'.format(sum_rewards))
                plt.pause(.001)

            # Process the state as a stack of three images
            next_stacked_state, next_stacked_frames = stack_frames(car.stacked_frames, next_state, False)
            car.stacked_frames = next_stacked_frames
            next_state = next_stacked_state

            if cur_step == 50:
                export_image(next_stacked_state, cur_step)

            # Set up the episode to be stored in the episode buffer
            episode = np.array([[state], action, reward, [next_state], done])
            episode = episode.reshape(1,-1)

            # Store the experience in the episode buffer
            episode_buffer.add(episode)

            # Update the running rewards
            sum_rewards += reward

            # Update the state
            state = next_state

        # Increment the episode counter
        num_episode += 1

        # If current step has not incremented then it failed
        # to load the episode
        if cur_step == 1:
            return

        # Dump the episode buffer to the main one
        car.experience_buffer.add(episode_buffer.buffer)
        rewards.append(sum_rewards)

        # Once the episode's finished, we proceed to train the network
        if num_episode % update_freq == 0:
            print('Entrenando la red...')

            for _ in range(num_epochs):
                loss = car.train(batch_size, supervised=True)
                losses.append(loss)

            # Update the target model with values from the main model
            car.update_target_network()

            print('Entrenamiento completado...')
            if num_episode % save_every == 0:
                # Save the model
                car.save_models()

        # Print the statistics
        if num_episode % print_every == 0:

            mean_loss = np.mean(losses[-(print_every * num_epochs):])

            print("Num episode: {} Mean reward: {:0.4f} Prob random: {:0.4f}, Loss: {:0.04f}".format(
                num_episode, np.mean(rewards[-print_every:]), car.epsilon, mean_loss))

    car.save_models()


def play(car, max_num_episodes, max_num_step, goal):

    car.epsilon = car.epsilon_min

    num_episode = 0
    while num_episode < max_num_episodes:

        # Get the game state from the environment
        state = env.reset()
        # Solves the bug that prevents gym from rendering
        # in 'state_pixels' mode
        env.env.viewer.window.dispatch_events()

        # Process the state as a stack of three images
        stacked_state, stacked_frames = stack_frames(car.stacked_frames, state, True)
        # Save the last frames used to produce the stack 
        # (they will be used to create the next one)
        car.stacked_frames = stacked_frames
        state = stacked_state

        # Whether the Game is complete or not
        done = False
        # Total reward obtained within the episode
        sum_rewards = 0
        # Current step of the episode
        cur_step = 0

        while cur_step < max_num_step and not done:

            cur_step += 1
            env.render()

            # Get the action to perform for the state
            action = car.get_action(state)

            # Perform the action and retrieve the next state, reward and done
            next_state, reward, done, _ = env.step(convert_action_to_gym(action))

            # Process the state as a stack of three images
            next_stacked_state, next_stacked_frames = stack_frames(car.stacked_frames, next_state, False)
            car.stacked_frames = next_stacked_frames
            next_state = next_stacked_state

            if cur_step == 50:
                export_image(next_stacked_state, cur_step)

            # Update the running rewards
            sum_rewards += reward

            # Update the state
            state = next_state

        # Increment the episode counter
        num_episode += 1
        rewards.append(sum_rewards)



# ----- Initialization ----- 

def setup(load_models=True):
    '''
    Clears Keras session and initializes the car agent

    Returns: a car agent
    '''

    # Reset Keras cache and any previous content in memory
    K.clear_session()

    # Setup our Agent
    car = CarAgent(load_models=load_models)
    
    # Print a summary of the model
    car.main_qn.summary()

    return car

if __name__ == "__main__":
    
    res = input('¿Quieres entrenar al coche sin supervisión (t), con supervisión (s) o jugar (p)?  \n')
    
    if 't' == res or 'T' == res:
        print('***** Unsupervised Training *****\n')
        
        # Whether you want to load previous saved weights 
        # or just start the training from the beginning.
        # Caution! If weights exist in the directory but
        # you decide to not load them, they will be overwritten 
        # by the new ones.
        load_models = True
        car = setup(load_models=load_models)

        # ----- Training hyperparameters ----- 

        # How many images to use in each training session our agent performs
        batch_size = 64
        # Number of epochs (# of forward and backward passes) to train
        num_epochs = 20
        # How often to perform an update on the networks' weights
        update_freq = 5

        # Number of steps needed to reduce epsilon -> epsilon_min
        annealing_steps = 1000.0
        # Maximum number of episodes allowed if we can't reach our goal
        max_num_episodes = 10000
        # Number of episodes in which only random actions will be taken
        # it is usually done at the beginning
        pre_train_episodes = 100
        # Episode's maximum length (we'll finish it if overpassed)
        max_num_step = 500

        # Training goal: If reached, the training will stop
        # and it will be considered as successful
        goal = 500

        # Start the training
        train(car, batch_size, num_epochs, update_freq, annealing_steps, 
        max_num_episodes, pre_train_episodes, max_num_step, goal)

    elif 'p' == res or 'P' == res:
        print('***** Play *****\n')
        
        load_models = True
        car = setup(load_models=load_models)

        # ----- Training hyperparameters ----- 

        max_num_episodes = 5
        max_num_step = 500
        goal = 500

        play(car, max_num_episodes, max_num_step, goal)
        print('Game completed!')

    elif 's' == res or 'S' == res:
        print('***** Supervised Training *****\n')
        
        # Whether you want to load previous saved weights 
        # or just start the training from the beginning.
        # Caution! If weights exist in the directory but
        # you decide to not load them, they will be overwritten 
        # by the new ones.
        load_models = True
        car = setup(load_models=load_models)

        # ----- Training hyperparameters ----- 

        # How many images to use in each training session our agent performs
        batch_size = 64
        # Number of epochs (# of forward and backward passes) to train
        num_epochs = 20
        # How often to perform an update on the networks' weights
        update_freq = 5
        # How many iterations over the set of recorded samples
        iterations = 30

        # Start the training
        for i in range(iterations):
            print('\n *** ITERATION {} OF {} *** \n'.format(i, iterations))
            train_s(car, batch_size, num_epochs, update_freq, verbose=False)