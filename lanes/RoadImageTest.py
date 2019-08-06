import numpy as np
import cv2
from RoadImage import RoadImage
import datetime

from debugtools import draw_analysed_image

if __name__ == "__main__":

    # Load a color image in grayscale
    img = cv2.imread('/home/tfg/Escritorio/ROI2.png', -1)
    
    ri = RoadImage(img)
    
    points = ri.analyse()
    raw_image = ri.get_image()
    
    draw_analysed_image(raw_image, points)

    print("Distance: {}".format(ri.center_offset()))
    print("RI {}".format(ri.center_offset()/90))


    cv2.imshow("Test", raw_image)
    cv2.imshow("Tes2t", ri.get_hough())

    cv2.waitKey(0)
    cv2.destroyAllWindows()