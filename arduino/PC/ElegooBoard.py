import serial

'''
Class that handles communications with the board 
'''

class ElegooBoard:

    '''
        By default, port and frequencies are hardcoded to match
        our own ElegooBoard and VM configuration. You may change
        this parameters as required.
    '''
    def __init__(self):

        self.board = None
        self.port = "/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_85735313233351D092A0-if00"
        self.freq = 9600
    
    '''
        Tries to open a connection between the PC and the board

        Return: Whether the connection was succesfully stablished 
        or not
    '''
    def open(self):

        res = False

        try:
            self.board = serial.Serial(self.port, self.freq)
            res = True
        except Exception as _:
                print("No se ha podido abrir una conexión con el puerto y la frecuencia indicadas:")
                print("\t - Puerto: \"{}\"".format(self.port))
                print("\t - Frecuencia: {} baudios".format(self.freq))
        
        return res
    
    """
        Closes a current open connection

        Return: If the connection was closed or not
    """
    def close(self):
        
        res = False

        if self.board:
            self.board.close()
            self.board = None
            res = True
        
        return res
    
    """
        Test whether there exists an open connection
        or not.
    """
    def isOpen(self):
        return self.board is not None
    

    """
        Sends an action to the board over an open connection

        Raises:

            - ValueException: If the provided action is not
            a valid NN Action

            - Exception: If the connection over which the action
            is trying to be sent is closed
    """
    def send_directions(self, action):

        #NN's Actions --> ['Izquierda', 'Centro', 'Derecha', 'Centro-Gas', 'Freno']

        if self.board:
            
            if action is not None and type(action) is int and action >= 0 and action < 5:
                self.board.write(str(action).encode())
                print("Se ha enviado la acción {}".format(action))
            else:
                raise ValueError("La acción proporcionada no es válida")

        else:
            raise Exception("Se ha intentado enviar una orden sobre una conexión cerrada")

