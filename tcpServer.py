from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import io
from io import BytesIO
import socket
import base64

#---------------------- CONSTANTES ----------------------

TCP_IP = '192.168.0.18'
#TCP_IP = '192.168.43.78'
TCP_PORT = 447
BUFFER_SIZE = 300000

#--------------------------------------------------------

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
    
    # Loop to process all the data received
    while 1:

        try:
            # We receive a message
            data_recv = conn.recv(BUFFER_SIZE)
            data_decoded = data_recv.decode()

            # Then split the images
            data_splited = data_decoded.split('*')
            # And get the second one
            data_1 = data_splited[1]
            
            # Break if we don't receive anything
            if not data_recv: 
                break

            sbuf = BytesIO()
            data_image = base64.b64decode(data_1)
            sbuf.write(data_image)

            pimg = Image.open(sbuf)

            # Conversion from PIL to OpenCV
            opencvImage = np.array(pimg)

            # Display the OpenCV image
            cv2.imshow('Demo Image',opencvImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Display the PIL image
            #pimg.show()
            
        except:
            continue

    conn.close()

if __name__ == '__main__':
    tcp_server()