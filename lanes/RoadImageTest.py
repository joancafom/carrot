import numpy as np
import cv2
from RoadImage import RoadImage
import datetime

if __name__ == "__main__":

    # Load an color image in grayscale
    img = cv2.imread('/home/tfg/Escritorio/ROI4.png', -1)
    
    ri = RoadImage(img)
    
    points = ri.analyse()
    raw_image = ri.get_image()
    last_point = None
    
    for p in points:

        if p is None:
            continue

        p_int = tuple([int(x) for x in p])
        print(p_int)
        cv2.circle(raw_image, p_int, 5, (255,0,0), 5)  

        if last_point:
            cv2.line(raw_image, last_point, p_int, (0, 255, 0), 3)

        last_point = p_int  

    print(ri.compute_distance_points(points[-1], points[-2]))

    print("RI {}".format(ri.center_offset()))


    cv2.imshow("Test", raw_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()