#!/usr/bin/env python

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
BUFFER_SIZE = 1000000  # Normally 1024, but we want fast response
 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)
print('Listening on {}:{}'.format(TCP_IP, TCP_PORT))
 
conn, addr = s.accept()
print('Accepted connection from:', addr)

file_completo = open("completo.txt", "w")
file_0 = open("primero.txt", "w")
  
while 1:
    try:
        data_recv = conn.recv(BUFFER_SIZE)
        #data_utf = data.decode("utf-8")
        data_decoded = data_recv.decode()
        file_completo.write(data_decoded)

        data_splited = data_decoded.split('****')
        data_0 = data_splited[1]
        file_0.write(data_0)
        print(data_0)
        #file.write(str(data_0))
        #print(data_0)
        '''if '-------------------------------------------'.encode() in data:
            continue
        missing_padding = len(data_0) % 4
        if missing_padding:
            data_0 += '='* (4 - missing_padding)
            print(data_0)
        file_completo.write(str(data_0))'''
        if not data_recv: 
            break

        sbuf = BytesIO()
        data_image = base64.b64decode(data_0)
        sbuf.write(data_image)
        pimg = Image.open(sbuf)
        #image = Image.open(io.BytesIO(data_image))
        #cv2.cvtColor(np.float32(pimg), cv2.COLOR_BGR2RGB)
        #cv2.imshow('image', pimg)
        pimg.show()
        #print("received data:", decodeddata)
        #conn.send(data)  # echo
    except:
        continue
conn.close()
file_completo.close()
file_0.close()