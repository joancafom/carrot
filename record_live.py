import tkinter as tk
from tkinter import Label
from tk_debouncer.debouncer import Debouncer

# Elegoo Board
from arduino.PC.ElegooBoard import ElegooBoard

# TCP Server
from tcpServer import tcp_server, car_env
import threading

# Export 
import numpy as np
import os
import csv
import datetime

# ----- Constants ----- 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATE_FORMAT = '%d%m%Y_%H%M%S'
SUMMARY_DELIMITER = '|'
FOLDER_NAME_FORMAT = 'pack_{}'
SUMMARY_NAME_FORMAT = 'summary.csv'
RECORD_MAIN_PATH = 'live_records/'
ACT_DESCRIPTIONS = [
        'Izquierda',
        'Centro',
        'Derecha',
        'Centro-Gas',
        'Freno'
]

'''
Records a real life driving session
using the camera sensor and the arduino 
board to control de RC Car.

Saves each frame in NumPy format,
as well as the actions that were taken.

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

live_records/pack_DDMMYYYY_hhmmss

        DDMMYYYY: DayMonthYear
        hhmmss: HoursMinutesSeconds
'''

class CarrotRecorder(object):

    # Creates a window that outputs the 
    # key that is being pressed at the moment.
    # Also sends the command via SerialPort 
    # so that the Arduino board can perform it.
    def __init__(self):
        
        # The actual window-app instance
        #
        # A debouncer is needed to solve an issue many OSs have:
        # they detect a single long press as multiple frequent
        # key presses/releases. Debouncer transforms that into 
        # a single key press & release
        #
        # ElegooBoard is an auxiliary class that handles
        # connections with arduino-like boards
        self.tkinterApp = tk.Tk()
        self.debouncer = Debouncer(self._pressed_cb, self._released_cb, self._nd_pressed_cb)
        self.board = ElegooBoard()

        # UI initialization
        self.label1 = Label(self.tkinterApp, text='None', width=50, bg='yellow')
        self.label1.pack()

        # Event Binding: What functions fire when a keyPress/Release occurs
        self.tkinterApp.bind('<KeyPress>', self.debouncer.pressed)
        self.tkinterApp.bind('<KeyRelease>', self.debouncer.released)

        # Last key pressed
        self.last_key = 'None'

        # Status Registers
        self.c_state = None
        self.c_next_state = None

        # Export
        self.steps = 0
        self.csv_writer = None
        self.summary_file = None
        self.gameplay_path = None

    # Executed multiple times while the 
    # key is being pressed (non-debounced press)
    # Sends the command to the board
    def _nd_pressed_cb(self, event):

        if event.keysym == 'a':

            self.c_state = car_env.get_state()
            self.board.send_directions(0)
            self.c_next_state = car_env.get_state()
            print('Left')

        elif event.keysym == 'd':

            self.c_state = car_env.get_state()
            self.board.send_directions(2)
            self.c_next_state = car_env.get_state()
            print('Right')

        elif event.keysym == 'w':

            self.c_state = car_env.get_state()
            self.board.send_directions(3)
            self.c_next_state = car_env.get_state()
            print('Up')

        elif event.keysym == 's':

            self.c_state = car_env.get_state()
            self.board.send_directions(4)
            self.c_next_state = car_env.get_state()
            print('Down')

    # Executed only one time while the 
    # key is being pressed (debounced press)
    # Updates the UI when the key is pressed
    def _pressed_cb(self, event):
        self.last_key = str(event.keysym)
        self.label1.config(text=self.last_key)

    # Executed only one time while the 
    # key is being pressed (debounced press)
    # Updates the UI when the key is released
    def _released_cb(self, event):
        if str(event.keysym) == self.last_key:
            self.label1.config(text='None')
            self.last_key = 'None'

    def a(self):

        if self.last_key is 'None':
            self.c_state = car_env.get_state()
            self.c_next_state = car_env.get_state()

        # Write the states and add a new line in the summary
        np.savez(self.savez_path.format(self.steps + 1), **{'01_state' : self.c_state, '02_next_state': self.c_next_state})
        self.csv_writer.writerow([self.last_key, 2, 0])
        # Asynchronous writing...
        self.summary_file.flush()

        print("Action: {}".format(self.last_key))
        print("S: {}".format(self.c_state))
        print("NS: {}".format(self.c_next_state))

        self.steps += 1
        self.tkinterApp.after(500, self.a)

    # Entry point of the application
    # Initializes connections and passes
    # mainloop to the TKinter app
    def load(self):
        self.board.open()
        self.tkinterApp.after(500, self.a)
        self.tkinterApp.mainloop()

def main():

    tkinterApp = CarrotRecorder()

    try:
        t = threading.Thread(target=tcp_server)
        t.start()
        input("Press any key to continue...")

    
    except Exception as e:
        print(e)

    # --- Create the folders where to record ---

    # First the main '/live_records' folder
    records_path = os.path.join(BASE_DIR, RECORD_MAIN_PATH)
    
    if not os.path.exists(records_path):
            os.makedirs(records_path)
    
    # Now the current gameplay folder
    today = datetime.datetime.today()
    today_str = today.strftime(DATE_FORMAT)
    tkinterApp.gameplay_path = os.path.join(records_path, FOLDER_NAME_FORMAT.format(today_str))
    
    if not os.path.exists(tkinterApp.gameplay_path):
            os.makedirs(tkinterApp.gameplay_path)

    # --- Create the summary files and begin to record ---
    summary_path = os.path.join(tkinterApp.gameplay_path, SUMMARY_NAME_FORMAT)
    tkinterApp.savez_path = os.path.join(tkinterApp.gameplay_path, '{}.npz')

    with open(summary_path, mode='w', encoding='utf-8') as summary:

            tkinterApp.summary_file = summary
            tkinterApp.csv_writer = csv.writer(tkinterApp.summary_file, delimiter=SUMMARY_DELIMITER)

            # Column description
            tkinterApp.csv_writer.writerow(['Action', 'Reward', 'Done'])

            # Begin the game
            tkinterApp.load()
    
    # Close the file's writing stream
    tkinterApp.summary_file.close()

    print('*** Live record was succesfully saved! *** \n')
    print('Path: {}'.format(tkinterApp.gameplay_path))


if __name__ == '__main__':
    main()