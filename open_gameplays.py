import os
import sys
import csv
import datetime
import numpy as np
from PIL import Image

#Import Constants and other expressions
from record_gameplay import SUMMARY_DELIMITER, SUMMARY_NAME_FORMAT, RECORD_MAIN_PATH, ACT_DESCRIPTIONS

"""
Creates a generator that extracts the 
information out of a given episode path.
If executed directly, provides a
convinient way to test recorded episodes
out by exporting state information as
images in a folder named 'export'. 
"""



# ----- Constants ----- 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_FOLDER_NAME = 'export'
IMAGES_PATH_FORMAT = None # Must be initialized if main is called



# ----- Auxiliary Functions ----- 

def export_image(state, seq):

    i = state.astype(np.uint8)
    imgRGB = Image.fromarray(i, 'RGB')
    imgRGB.save(IMAGES_PATH_FORMAT.format(seq))



# ----- Main Functionality ----- 

def open_episode(episode_path):

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

        csv_reader = csv.reader(summary, delimiter=SUMMARY_DELIMITER)
            
        current_line = -1
        total_reward = 0
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
                total_reward += reward

                yield state, next_state, action, reward, done

            except Exception as e:
                print('\n Oops... Parece que el directorio está corrupto. Pasando a la siguiente línea \n')
                print('\n *** TRAZA DE LA EXCEPCIÓN *** \n')
                print(str(e))
                continue
            
        print('\n Total Reward of the Episode: {:0.4f} \n'.format(total_reward))
                    
    summary.close()



# ----- Test Export ----- 

if __name__ == "__main__":

    # Ask the user for the name of the episode folder
    folder_name = input('\n Escriba el nombre de la carpeta donde se encuentra el episodio (pack): \n')

    # Construct the path to the folder where the episode was recorded
    episode_path = os.path.join(BASE_DIR, RECORD_MAIN_PATH, folder_name)

    # Create the folder where to export the images 
    # only if the specified path does exist
    export_path = os.path.join(episode_path, EXPORT_FOLDER_NAME)
    if os.path.exists(episode_path) and not os.path.exists(export_path):
        os.makedirs(export_path)
    
    IMAGES_PATH_FORMAT = os.path.join(export_path, 'export_{}.png')

    print('\n Trabajando...')
    i = 0
    for state, next_state, action, reward, done in open_episode(episode_path):
        i += 1
        export_image(state, i)
        export_image(next_state, (str(i) + '_next'))