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

    bottom_center = (w/2, h//2-20)
    left_point = None
    left_point_partner = None
    left_point_distance = None
    right_point = None
    right_point_partner = None
    right_point_distance = None


    # Obtenemos los dos puntos más cercanos al bottom-center
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Los dos puntos que definen el segmento
        puntos_linea = [(x1, y1), (x2, y2)]

        i = 0
        for punto in puntos_linea:
            
            i += 1
            # Vector que une el punto con el centro
            d_centro_punto = (punto[0] - bottom_center[0], punto[1] - bottom_center[1])
            distance = np.linalg.norm(d_centro_punto)

            if bottom_center[0] >= punto[0]:
                # Se encuentra a la izq
                if left_point is None or left_point_distance > distance:
                    left_point = punto
                    left_point_partner = puntos_linea[(i%len(puntos_linea))]
                    left_point_distance = distance
            else:
                # Se encuentra a la dch
                if right_point is None or right_point_distance > distance:
                    right_point = punto
                    right_point_partner = puntos_linea[(i%len(puntos_linea))]
                    right_point_distance = distance
        
        
        cv2.circle(roi, (x1, y1), 2, (255, 0, 0), 3)
        cv2.circle(roi, (x2, y2), 2, (255, 0, 0), 3)
    
    cv2.circle(roi, left_point, 3, (0, 255, 0), 3)
    cv2.circle(roi, right_point, 3, (0, 255, 0), 3)
    cv2.circle(roi, left_point_partner, 3, (0, 0, 255), 3)
    cv2.circle(roi, right_point_partner, 3, (0, 0, 255), 3)

    def get_pendiente(x1, y1, x2, y2):
        print(x1, y1, x2, y2)
        if x2-x1 != 0:
            return (y2-y1)/(x2-x1)
        else:
            return (y2-y1)
    
    def get_independiente(x1, y1, x2, y2):
        
        if x2-x1 != 0:
            return (x1*(y1-y2))/(x2-x1) + y1
        else:
            return x1*(y1-y2) + y1
    
    left_m = get_pendiente(left_point[0], left_point[1], left_point_partner[0], left_point_partner[1])
    right_m = get_pendiente(right_point[0], right_point[1], right_point_partner[0], right_point_partner[1])

    left_n = get_independiente(left_point[0], left_point[1], left_point_partner[0], left_point_partner[1])
    right_n = get_independiente(right_point[0], right_point[1], right_point_partner[0], right_point_partner[1])

    corte_x = (left_n - right_n) / (right_m - left_m)
    corte_y = left_m * corte_x + left_n

    punto_corte = (int(corte_x), int(corte_y))
    bottom_int = (int(bottom_center[0]), int(bottom_center[1]))

    cv2.line(roi, punto_corte, bottom_int, (255, 255, 0), 3)
    cv2.circle(roi, bottom_int, 3, (0, 0, 255), 3)

    top_left_point = - left_n // left_m
    bottom_left_point = ((h//2-20) - left_n) // left_m
    top_right_point = - right_n // right_m
    bottom_right_point = ((h//2-20) - right_n) // right_m
    cv2.circle(roi, (int(top_left_point), 0), 3, (0, 0, 255), 3)
    cv2.circle(roi, (int(bottom_left_point), h//2-20), 3, (0, 0, 255), 3)
    cv2.line(roi, (int(top_left_point), 0), (int(bottom_left_point), h//2-20), (255, 255, 0), 3)
    cv2.circle(roi, (int(top_right_point), 0), 3, (0, 0, 255), 3)
    cv2.circle(roi, (int(bottom_right_point), h//2-20), 3, (0, 0, 255), 3)
    cv2.line(roi, (int(top_right_point), 0), (int(bottom_right_point), h//2-20), (255, 255, 0), 3)

    def get_incentro(punto_corte, punto_left, punto_right):

        d_pc_left = [punto_left[0] - punto_corte[0], punto_left[1] - punto_corte[1]]
        d_pc_left_module = np.linalg.norm(d_pc_left)
        d_pc_right = [punto_right[0] - punto_corte[0], punto_right[1] - punto_corte[1]]
        d_pc_right_module = np.linalg.norm(d_pc_right)
        d_left_right = [punto_left[0] - punto_right[0], punto_left[1] - punto_right[1]]
        d_left_right_module = np.linalg.norm(d_left_right)

        sum_modules = d_pc_left_module + d_pc_right_module + d_left_right_module

        x_i = (punto_corte[0]*d_left_right_module + punto_right[0]*d_pc_left_module + punto_left[0]*d_pc_right_module) / sum_modules
        y_i = (punto_corte[1]*d_left_right_module + punto_right[1]*d_pc_left_module + punto_left[1]*d_pc_right_module) / sum_modules

        return (x_i, y_i)
    
    incentro = get_incentro(punto_corte, left_point, right_point)
    cv2.circle(roi, (int(incentro[0]), int(incentro[1])), 3, (255, 0, 255), 3)

    bisectriz_m = get_pendiente(incentro[0], incentro[1], punto_corte[0], punto_corte[1])
    bisectriz_n = get_independiente(incentro[0], incentro[1], punto_corte[0], punto_corte[1])
    bottom_bisectriz = ((h//2-20) - bisectriz_n) // bisectriz_m
    cv2.circle(roi, (int(bottom_bisectriz), h//2-20), 3, (0, 0, 255), 3)
    cv2.line(roi, punto_corte, (int(bottom_bisectriz), h//2-20), (255, 0, 255), 3)


    cv2.imshow("Hough", roi)
    cv2.waitKey(0)

    

                




    
