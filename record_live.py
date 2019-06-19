from arduino.PC.ElegooBoard import ElegooBoard
from tk_debouncer.debouncer import Debouncer
import tkinter as tk
from tkinter import Label

'''
Records a real life driving session
using the camera sensor and the arduino 
board to control de RC Car.
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

    # Executed multiple times while the 
    # key is being pressed (non-debounced press)
    # Sends the command to the board
    def _nd_pressed_cb(self, event):

        if event.keysym == 'Left':
            self.board.send_directions(0)
            print('Left')
        elif event.keysym == 'Right':
            self.board.send_directions(2)
            print('Right')
        elif event.keysym == 'Up':
            self.board.send_directions(3)
            print('Up')
        elif event.keysym == 'Down':
            self.board.send_directions(4)
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

    # Entry point of the application
    # Initializes connections and passes
    # mainloop to the TKinter app
    def load(self):
        self.board.open()
        self.tkinterApp.mainloop()

def main():
    tkinterApp = CarrotRecorder()
    tkinterApp.load()

if __name__ == '__main__':
    main()