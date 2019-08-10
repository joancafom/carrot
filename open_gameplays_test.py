from record_gameplay import BASE_DIR, RECORD_MAIN_PATH
from open_gameplays import open_episode
import cv2
import os

from lanes.RoadImage import RoadImage
from debugtools import draw_analysed_image

'''
        Tests both the function to open gameplays
        and the gameplay itself, by showing the frames,
        the result of the analysis and the obtained
        reward. Aditionally, at the end provides a summary
        of all the rewards per episode.
'''

records_path = os.path.join(BASE_DIR, RECORD_MAIN_PATH)
actions = ['Izquierda', 'Centro', 'Derecha', 'Centro-Gas', 'Freno']
    
num_episode = 0
rewards_summary = {}

for direc in os.listdir(records_path):

    num_episode += 1
    
    print('\n--- Episode {} - {}'.format(num_episode, direc))

    episode_path = os.path.join(records_path, direc)
    ep_acc_reward = 0
    for e_state, e_next_state, e_action, e_reward, e_done in open_episode(episode_path):

        ri = RoadImage(e_state)
    
        points = ri.analyse()
        raw_image = ri.get_image()
        
        draw_analysed_image(raw_image, points)

        cv2.imshow("Analysis", raw_image)
        cv2.waitKey(1)
        print(actions[e_action])

        ep_acc_reward += e_reward
        rewards_summary[episode_path] = ep_acc_reward

print('---- Episodes summary ----- \n')
for k,v in rewards_summary.items():
        print('\t {}: {}'.format(k,v))
