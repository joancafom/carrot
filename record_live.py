from arduino.PC.ElegooBoard import ElegooBoard
from tk_debouncer.debouncer import Debouncer
import tkinter as tk
from tkinter import Label

class CarrotRecorder(object):

    def __init__(self):
        self.tkinterApp = tk.Tk()
        self.debouncer = Debouncer(self._pressed_cb, self._released_cb, self._nd_pressed_cb)
        self.board = ElegooBoard()

        # UI initialization
        self.label1 = Label(self.tkinterApp, text='None', width=50, bg='yellow')
        self.label1.pack()

        # Event Binding
        self.tkinterApp.bind('<KeyPress>', self.debouncer.pressed)
        self.tkinterApp.bind('<KeyRelease>', self.debouncer.released)

        self.last_key = 'None'

    def _nd_pressed_cb(self, event):

        # Send the command
        if event.keysym == 'Left':
            #self.board.send_directions(0)
            print('Left')
        elif event.keysym == 'Right':
            #self.board.send_directions(2)
            print('Right')
        elif event.keysym == 'Up':
            #self.board.send_directions(3)
            print('Up')
        elif event.keysym == 'Down':
            #self.board.send_directions(4)
            print('Down')

    def _pressed_cb(self, event):
        self.last_key = str(event.keysym)
        self.label1.config(text=self.last_key)

    def _released_cb(self, event):
        if str(event.keysym) == self.last_key:
            self.label1.config(text='None')
            self.last_key = 'None'

    def load(self):
        self.board.open()
        self.tkinterApp.mainloop()

def main():
    tkinterApp = CarrotRecorder()
    tkinterApp.load()

if __name__ == '__main__':
    main()