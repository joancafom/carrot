from tk_debouncer.debouncer import Debouncer
import tkinter as tk
from tkinter import Label

class CarrotRecorder(object):

    def __init__(self):
        self.app = tk.Tk()
        self.debouncer = Debouncer(self._pressed_cb, self._released_cb)
        self.app.bind('<KeyPress>', self.debouncer.pressed)
        self.app.bind('<KeyRelease>', self.debouncer.released)
        self.pressedKey = None

        self.label1 = Label(self.app, text='prompt', width=50, bg='yellow')
        self.label1.pack()
        self.lt = 'None'


    def _pressed_cb(self, event):
        #print('Pressed!: {}'.format(event.keysym))
        self.lt = str(event.keysym)
        self.label1.config(text=self.lt)
        #print(self.lt)



    def _released_cb(self, event):
        #print('Released!: {}'.format(self.lt))
        if str(event.keysym) == self.lt:
                self.label1.config(text='None')
                self.lt = 'None'


    def loop(self):
        self.app.mainloop()


def main():
    app = CarrotRecorder()
    app.loop()


if __name__ == '__main__':
    main()