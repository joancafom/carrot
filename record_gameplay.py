from pyglet.window import key
from gym.envs.box2d.car_racing import CarRacing
import numpy as np

# Gameplay Export
from _aux import convert_action_to_nn
import os
import csv
import datetime


'''
Records an entire gameplay and saves its
frames in NumPy format as well as the actions
that were taken.

For each episode, the program creates a folder
where to save the content relevant to just THAT 
episode. Inside the folder, you can find 
several .npz files containing both the actual 
frame and the following one. Also, a single .csv
with a summary of what is happenning.

Each line in the csv references to a .npz file.
I.E. the first line in .csv refers to 1.npz
     the second line in .csv refers to 2.npz
     and so on...

Folders are named using the following pattern:

records/pack_DDMMYYYY_hhmmss

        DDMMYYYY: DayMonthYear
        hhmmss: HoursMinutesSeconds
'''


# ----- Constants ----- 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATE_FORMAT = '%d%m%Y_%H%M%S'
SUMMARY_DELIMITER = '|'
FOLDER_NAME_FORMAT = 'pack_{}'
SUMMARY_NAME_FORMAT = 'summary.csv'
RECORD_MAIN_PATH = 'records/'
ACT_DESCRIPTIONS = [
        'Izquierda',
        'Centro',
        'Derecha',
        'Izq Gas',
        'Centro Gas',
        'Dcha Gas',
        'Freno'
]


def play(csv_writer, summary_file, episode_path):
        
        # ----- Keys Detection ----- 

        a = np.array( [0.0, 0.0, 0.0] )

        def key_press(k, mod):
                if k==key.LEFT:  a[0] = -1.0
                if k==key.RIGHT: a[0] = +1.0
                if k==key.UP:    a[1] = +1.0
                if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
        def key_release(k, mod):
                if k==key.LEFT  and a[0]==-1.0: a[0] = 0
                if k==key.RIGHT and a[0]==+1.0: a[0] = 0
                if k==key.UP:    a[1] = 0
                if k==key.DOWN:  a[2] = 0


        # ----- Game Setup ----- 
       
        env = CarRacing()
        env.render()

        env.viewer.window.on_key_press = key_press
        env.viewer.window.on_key_release = key_release

        # ----- Game Loop ----- 
        state = env.reset()
        total_reward = 0.0
        steps = 0
        savez_path = os.path.join(episode_path, '{}.npz')
        while True:

                # Convert the action from keyboard to nn's format
                a_nn = convert_action_to_nn(a)
                print('{} - {}'.format(a_nn, ACT_DESCRIPTIONS[a_nn]))

                next_state, reward, done, info = env.step(a)

                # Write the states and add a new line in the summary
                np.savez(savez_path.format(steps + 1), **{'01_state' : state, '02_next_state': next_state})
                csv_writer.writerow([a_nn, reward, done])
                # Asynchronous writing...
                summary_file.flush()

                total_reward += reward
                print(type(reward))
                if steps % 200 == 0 or done:
                        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                        print("step {} total_reward {:+0.2f}".format(steps, total_reward))

                steps += 1
                env.render()
                state = next_state

                if done: break
        env.close()


if __name__ == "__main__":
    
        # --- Create the folders where to record ---

        # First the main '/records' folder
        records_path = os.path.join(BASE_DIR, RECORD_MAIN_PATH)
        
        if not os.path.exists(records_path):
                os.makedirs(records_path)
        
        # Now the current gameplay folder
        today = datetime.datetime.today()
        today_str = today.strftime(DATE_FORMAT)
        gameplay_path = os.path.join(records_path, FOLDER_NAME_FORMAT.format(today_str))
        
        if not os.path.exists(gameplay_path):
                os.makedirs(gameplay_path)

        # --- Create the summary files and begin to record ---
        summary_path = os.path.join(gameplay_path, SUMMARY_NAME_FORMAT)

        with open(summary_path, mode='w', encoding='utf-8') as summary:

                csv_writer = csv.writer(summary, delimiter=SUMMARY_DELIMITER)

                # Column description
                csv_writer.writerow(['Action', 'Reward', 'Done'])

                # Begin the game
                play(csv_writer, summary, gameplay_path)
        
        # Close the file's writing stream
        summary.close()

        print('*** Gameplay was succesfully saved! *** \n')
        print('Path: {}'.format(gameplay_path))