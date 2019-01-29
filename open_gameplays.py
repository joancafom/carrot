import os
import sys
import csv
import datetime
import numpy as np
from PIL import Image

#Import Constants and other expressions
from record_gameplay import SUMMARY_DELIMITER, SUMMARY_NAME_FORMAT, RECORD_MAIN_PATH, ACT_DESCRIPTIONS

"""
Reads a pack folder and extracts the
information out of it. Optionally, it
exports the images to a folder named 
'export' in order to check them out.

The bool constant EXPORT_IMAGES determines
if the images are exported or not.
"""



# ----- Constants ----- 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_IMAGES = True
EXPORT_FOLDER_NAME = 'export'
IMAGES_PATH_FORMAT = ''


def export_image(state, seq):

    i = state.astype(np.uint8)
    imgRGB = Image.fromarray(i, 'RGB')
    imgRGB.save(IMAGES_PATH_FORMAT.format(seq))

if __name__ == "__main__":

        # Ask the user for the name of the episode folder
        folder_name = input('\n Escriba el nombre de la carpeta donde se encuentra el episodio (pack): \n')

        # Construct the path to the folder where the episode was recorded
        episode_path = os.path.join(BASE_DIR, RECORD_MAIN_PATH, folder_name)

        # Check the existance of the specified path
        if not os.path.exists(episode_path):
            sys.exit('\n Oops... Parece que la carpeta que ha especificado no existe')
        
        # Check the existance of summary.csv
        summary_path = os.path.join(episode_path, SUMMARY_NAME_FORMAT)
        if not os.path.exists(summary_path):
            sys.exit('\n Oops... Parece que el fichero resumen {} no existe en la carpeta especificada'.format(SUMMARY_NAME_FORMAT))

        # Create the folder where to export the images
        export_path = os.path.join(episode_path, EXPORT_FOLDER_NAME)
        if not os.path.exists(export_path):
                os.makedirs(export_path)
        
        IMAGES_PATH_FORMAT = os.path.join(export_path, 'export_{}.png')

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
                        if EXPORT_IMAGES:
                            export_image(states['01_state'], current_line)
                            export_image(states['02_next_state'], (str(current_line) + '_next'))

                        action = int(l[0])
                        reward = float(l[1])
                        done = l[2]
                        total_reward += reward

                        print('Frame {}: {} ({}) - Reward {:0.4f} - Done {}'.format(current_line, 
                        ACT_DESCRIPTIONS[action], action, reward, done))

                    except Exception as e:
                        print('\n Oops... Parece que el directorio está corrupto. Pruebe con otro directorio. \n')
                        print('\n *** TRAZA DE LA EXCEPCIÓN *** \n')
                        print(str(e))
                        sys.exit()
                
                print('\n Total Reward of the Episode: {:0.4f} \n'.format(total_reward))
                       
        summary.close()