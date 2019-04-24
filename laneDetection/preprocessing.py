import cv2
import numpy as np
import math

for c in range(1,12):
    image = cv2.imread("./base_imagenes/bd_{}.jpg".format(c))
    (h, w, d) = image.shape
    print("alto: {}, ancho: {}, profundidad: {}".format(h, w, d))
    #cv2.imshow("Imagen", image)
    #cv2.waitKey(0)

    # (B, G, R) = image[19, 89]
    # print("Blue: {}, Green: {}, Red: {}".format(B, G, R))

    roi = image[h//2:h-20, 0:w]
    #cv2.imshow("ROI", roi)
    #cv2.waitKey(0)

    # resized = cv2.resize(image, (200, 200))
    # #cv2.imshow("Resized", resized)
    # #cv2.waitKey(0)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Gray", gray)
    #cv2.waitKey(0)

    thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow("Thresh", thresh)
    #cv2.waitKey(0)

    edged = cv2.Canny(gray, 50, 150)
    #cv2.imshow("Edged", edged)
    #cv2.waitKey(0)

    edged2 = cv2.Canny(thresh, 50, 150)
    #cv2.imshow("Edged2", edged2)
    #cv2.waitKey(0)

    lines = cv2.HoughLinesP(edged2, 1, np.pi/100, 70, minLineLength=5, maxLineGap=10)
    
    def yMax(lista):

        p1 = (lista[0][0], lista[0][1])
        p2 = (lista[0][2], lista[0][3])

        if p1[1] > p2[1]:
            return p1[1]
        elif p1[1] == p2[1] and p1[0] < p2[0]:
            return p1[1]
        else:
            return p2[1]

    puntos_ordenados = sorted(lines, key=lambda x: yMax(x), reverse=True)
    res = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        p1 = (x1, y1)
        p2 = (x2, y2)

        cv2.circle(roi, (x1, y1), 3, (0, 0, 255), 3)
        cv2.circle(roi, (x2, y2), 3, (0, 255, 0), 3)

    for line in puntos_ordenados:
        x1, y1, x2, y2 = line[0]
        p1 = (x1, y1)
        p2 = (x2, y2)

        punto_inferior = None

        if not res:
            res.append(line)
            cv2.line(roi, (x1, y1), (x2, y2), (255, 0, 0), 3)
            continue

        if p1[1] > p2[1]:
            punto_inferior = p1
        elif p1[1] == p2[1] and p1[0] < p2[0]:
            punto_inferior = p1
        else:
            punto_inferior = p2

        entra = True
        c = 0
        print(" ----- Nueva linea -----")
        for elem in res:   
            c += 1
            e = elem[0]
            dr = (e[2] - e[0], e[3] - e[1])
            d_a_punto = (p2[0]- p1[0], p2[1]- p1[1])

            cos = np.dot(dr,d_a_punto) / (np.linalg.norm(dr) * np.linalg.norm(d_a_punto))
            cos_a = np.absolute(cos)
            print("Cos_a ", cos_a)
            angle = math.degrees(math.acos(cos))
            print("ANGLE {}".format(angle))
            print("Recta {}".format(e))
            print("dr ", dr)
            print("dp ", d_a_punto)
            print("Punto {}".format(punto_inferior))
            print("DOT ", np.dot(dr,d_a_punto))
            print("Cos_a".format(punto_inferior), cos_a)

            if cos_a >= 0.949 or len(res) >= 4:
                print("NO ENTRAAAAA Cos_a".format(punto_inferior), cos_a)
                entra = entra and False
                print("\t entra: {}".format(entra))
            
            if c == (len(res)):
                print('wig')
                print("\t entra: {}".format(entra))


        if entra:
            # print("Recta {}".format(e))
            # print("dr ", dr)
            # print("dp ", d_a_punto)
            # print("Punto {}".format(punto_inferior))
            # print("DOT ", np.dot(dr,d_a_punto))
            # print("Cos_a".format(punto_inferior), cos_a)
            res.append(line)

        #cv2.line(roi, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # if minor is None:
        #     minor = [x1, y1]
        
        # if minor[1] < y1 :
        #     minor = [x1, y1]
        # if minor[1] < y2:
        #     minor = [x2, y2]

    #cv2.circle(roi, (minor[0], minor[1]), 3, (255, 255, 0), 3)  
    #print(points)
    print("RES ", len(res))
    for line in res:
        x1, y1, x2, y2 = line[0]
        p1 = (x1, y1)
        p2 = (x2, y2)
        cv2.line(roi, (x1, y1), (x2, y2), (2, 166, 249), 3)
        cv2.imshow("Hough {}".format(len(lines)), roi)
        cv2.waitKey(0)
    
    
