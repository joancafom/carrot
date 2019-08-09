from record_gameplay import BASE_DIR, RECORD_MAIN_PATH
from open_gameplays import open_episode
import cv2
import os

from lanes.RoadImage import RoadImage
from debugtools import draw_analysed_image

records_path = os.path.join(BASE_DIR, 'live_records')
actions = ['Izquierda', 'Centro', 'Derecha', 'Centro-Gas', 'Freno']

    
num_episode = 0
d = {}

for direc in os.listdir(records_path):

    # Current step of the episode
    cur_step = 0

    num_episode += 1
    
    print('\n--- Episode {} - {}'.format(num_episode, direc))

    episode_path = os.path.join(records_path, direc)
    total = 0
    for e_state, e_next_state, e_action, e_reward, e_done in open_episode(episode_path):

        cur_step += 1

        ri = RoadImage(e_state)
    
        points = ri.analyse()
        raw_image = ri.get_image()
        
        draw_analysed_image(raw_image, points)
        cv2.imshow("Test", raw_image)



        cv2.waitKey(1)
        print(actions[e_action])
        total += e_reward
        print(num_episode)
        d[num_episode] = total
        #print("Episode ({}) and Reward {}".format(cur_step, total))

print(d)
