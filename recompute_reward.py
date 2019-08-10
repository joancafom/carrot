import os
import sys
import csv
import datetime
import numpy as np
from PIL import Image

#Import Constants and other expressions
from record_gameplay import SUMMARY_DELIMITER, SUMMARY_NAME_FORMAT, RECORD_MAIN_PATH, ACT_DESCRIPTIONS

from RDCentre import RDCentre

"""
    Computes again the reward of a recorded episode. In the eventual case
    that either the algorithm or the reward function changes, we can keep
    the same recordings that we have.

    Creates a new summary file named "summary.csv2" with the new rewards.
"""


# ----- Constants ----- 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----- Main Functionality ----- 

def recompute(episode_path):

    # Check the existance of the specified path
    if not os.path.exists(episode_path):
        print('\n Oops... Parece que la carpeta que ha especificado no existe')
        return
    
    # Check the existance of summary.csv
    summary_path = os.path.join(episode_path, SUMMARY_NAME_FORMAT)
    if not os.path.exists(summary_path):
        print('\n Oops... Parece que el fichero resumen {} no existe en la carpeta especificada'.format(SUMMARY_NAME_FORMAT))
        return
    
    # Open the episode
    with open(summary_path, mode='r', encoding='utf-8') as summary:
        with open(summary_path+'2', mode='w', encoding='utf-8') as new_summary:

            csv_reader = csv.reader(summary, delimiter=SUMMARY_DELIMITER)
            csv_writer = csv.writer(new_summary, delimiter=SUMMARY_DELIMITER)

            # Column description
            csv_writer.writerow(['Action', 'Reward', 'In Progress'])
                
            current_line = -1
            total_reward_new = 0
            total_reward_old = 0

            # Creates a new instance of RDCentre to ensure
            # independance between episodes
            rdCentre = None
            rdCentre = RDCentre()
            rdCentre.action_memory.clear()
            rdCentre.out_counter = 0
            rdCentre.reward_counter = 0
            

            for l in csv_reader:
                    
                current_line += 1
                    
                # The first line only contains the csv descriptions
                if current_line == 0:
                    continue

                try:
                    states_file = os.path.join(episode_path, '{}.npz'.format(current_line))
                    states = np.load(states_file)
                        
                    # state is under key '01_state'
                    # next_state is under key '02_next_state'
                    state = states['01_state']
                    next_state = states['02_next_state']

                    action = int(l[0])
                    reward = float(l[1])
                    done = l[2]

                    new_reward = rdCentre.get_reward(state, action)
                    csv_writer.writerow([action, new_reward, done])

                    total_reward_old += reward
                    total_reward_new += new_reward

                    new_summary.flush()


                except Exception as e:
                    print('\n Oops... Parece que el directorio está corrupto. Pasando a la siguiente línea \n')
                    print('\n *** TRAZA DE LA EXCEPCIÓN *** \n')
                    print(str(e))
                    continue
            
        print('\n Total Reward of the Episode: {:0.4f} (New) vs. {:0.4f} (Old) \n'.format(total_reward_new, total_reward_new))
                    
    summary.close()
    new_summary.close()



# ----- Test Export ----- 

if __name__ == "__main__":

    records_path = os.path.join(BASE_DIR, RECORD_MAIN_PATH)
    
    for direc in os.listdir(records_path):

        episode_path = os.path.join(records_path, direc)
        print("Recomputing - " + episode_path + "\n")
        
        recompute(episode_path)