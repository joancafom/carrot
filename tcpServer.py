from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import cv2

import io
from io import BytesIO
import socket
import base64

class Enviroment: 
    def __init__(self, state): 
         self._state = state

    def get_state(self): 
        return self._state
        
    def set_state(self, x): 
        self._state = x 

# Current status global variable

car_env = Enviroment("")

#---------------------- CONSTANTES ----------------------

TCP_IP = '10.0.2.15'
#TCP_IP = '192.168.0.18'
#TCP_IP = '192.168.43.78'
TCP_PORT = 447
BUFFER_SIZE = 300000

#--------------------------------------------------------

#-------------------- THREAD FUNCTION -------------------
def tcp_server():
    
    # Socket initialization
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))

    # We only want one connection
    s.listen(1)

    print('Listening on {}:{}'.format(TCP_IP, TCP_PORT))
    
    # The server starts listening
    conn, addr = s.accept()

    print('Accepted connection from:', addr)

    s_image = ""
    
    # Loop to process all the data received
    while 1:

        try:
            # We receive a message
            data_recv = conn.recv(BUFFER_SIZE)
            data_decoded = data_recv.decode()

            for c in data_decoded:
                
                # The "*" character delimits the images
                if c is not '*':
                    s_image += c

                elif s_image is not "":

                    sbuf = BytesIO()
                    data_image = base64.b64decode(s_image)
                    sbuf.write(data_image)

                    pimg = Image.open(sbuf)

                    # Conversion from PIL to OpenCV
                    opencvImage = np.array(pimg)

                    # Fix width to 200px
                    dim = (268, 200)
                    resizedI = cv2.resize(opencvImage, dim)

                    # Rotate the image
                    rotated90 = cv2.rotate(resizedI, rotateCode=0)
                    #cv2.imshow("rotated", rotated90)

                    # Crop image so its witdh is 200px and its height is 134px
                    roi = rotated90[dim[0]//2:(dim[0]//2)+134, 0:200]
                    #cv2.imshow("ROI", roi)
                    #cv2.waitKey(1)

                    s_image = ""

                    car_env.set_state(roi)

                    # Display the PIL image
                    #pimg.show()
            
        except Exception as e:
            print(e)
            continue

    conn.close()

#--------------------------------------------------------

if __name__ == '__main__':
    tcp_server()
