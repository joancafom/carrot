import threading
import sys
from tcpServer import tcp_server, car_env

# Step function
import cv2
import time
from collections import deque
from arduino.PC.ElegooBoard import ElegooBoard
from lanes.RoadImage import RoadImage
from lanes.debugtools import draw_analysed_image

'''
    The purpose of this class is to provide a standard point
    of access to external communication as well as duties in which
    the real driving is required. 

'''

ACTION_MEMORY_MAX = 8
MIDLANE_PX = 190/2

class RDCentre:

    # ------- COMMON ------- #
    board = ElegooBoard()

    # ------- STEP -------- #
    action_memory = deque([], maxlen=ACTION_MEMORY_MAX)
    out_counter = 0
    reward_counter = 0

    def initialize(self):

        self.board.open()
        
        try:

            t = threading.Thread(target=tcp_server)
            t.start()
    
        except Exception as e:
            print("\n\n FATAL: Could not start TCP Server. Exiting \n\n")
            print(e)
            self.board.close()
            sys.exit()
    
    def get_road_picture(self):
        return car_env.get_state()

    def perform_step(self, action):

        step_reward = 0
        done = False
        
        # 1. The car performs the action and we wait for
        # it to happen

        self.board.send_directions(int(action))
        #time.sleep(0.25)

        state = self.get_road_picture()

        # We keep track of the actions we've taken, 
        # to ensure that the reward is updated accordingly.
        self.action_memory.append(int(action))

        # Analyse the image obtained from the phone
        ri = RoadImage(state)
        points = ri.analyse()
        raw_image = ri.get_image()

        draw_analysed_image(raw_image, points)

        # For debugging purposes, we show it
        cv2.imshow("PhoneCam", raw_image)
        cv2.waitKey(1)

        # Get the distance to the midlane to compute the reward
        midlane_distance = ri.center_offset()

        # If the distance could not be computed or it exceeds the width
        # of one of our lanes
        if not midlane_distance or midlane_distance > MIDLANE_PX:

            if self.out_counter >= 8:
                done = False
                step_reward = -100
                self.out_counter = 0

                print(" You've been kicked out")
            else:
                self.out_counter += 1
        
        else:

            self.out_counter = 0

            # If the car has not moved forward in 
            # the last 8 frames, we set the reward
            # to be -2.5
            if 3 not in self.action_memory:
                step_reward = -2.5
            else:
                norm_distance = midlane_distance / MIDLANE_PX
                step_reward += 5 * (1 - norm_distance)
        
        self.reward_counter += step_reward

        print("Step Reward: {}".format(self.reward_counter))

        return state, step_reward, done
    
    def get_reward(self, state, action):

        step_reward = 0

        self.action_memory.append(int(action))

        # Analyse the image obtained from the phone
        ri = RoadImage(state)
        points = ri.analyse()
        raw_image = ri.get_image()

        draw_analysed_image(raw_image, points)

        # For debugging purposes, we show it
        cv2.imshow("PhoneCam", raw_image)
        cv2.waitKey(1)

        # Get the distance to the midlane to compute the reward
        midlane_distance = ri.center_offset()

        # If the distance could not be computed or it exceeds the width
        # of one of our lanes
        if not midlane_distance or midlane_distance > MIDLANE_PX:

            if self.out_counter >= 8:
                step_reward = -100
                self.out_counter = 0

                print(" You've been kicked out")
            else:
                self.out_counter += 1
        
        else:

            self.out_counter = 0

            # If the car has not moved forward in 
            # the last 8 frames, we set the reward
            # to be -2.5
            if 3 not in self.action_memory:
                step_reward = -2.5
            else:
                norm_distance = midlane_distance / MIDLANE_PX
                step_reward += 5 * (1 - norm_distance)
        
        self.reward_counter += step_reward

        print("Step Reward: {}".format(self.reward_counter))

        return step_reward