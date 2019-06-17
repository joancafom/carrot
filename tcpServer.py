from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import io
from io import BytesIO
import socket
import base64

TCP_IP = '192.168.0.18'
#TCP_IP = '192.168.43.78'
TCP_PORT = 447
BUFFER_SIZE = 300000
 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)
print('Listening on {}:{}'.format(TCP_IP, TCP_PORT))
 
conn, addr = s.accept()
print('Accepted connection from:', addr)

#file_completo = open("completo.txt", "w")
#file_1 = open("primero.txt", "w")
  
while 1:

    try:
        data_recv = conn.recv(BUFFER_SIZE)
        data_decoded = data_recv.decode()

        #file_completo.write(data_decoded)

        data_splited = data_decoded.split('****')
        data_1 = data_splited[1]
        #file_1.write(data_1)
        #print(data_1)
        
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
#file_completo.close()
#file_1.close()