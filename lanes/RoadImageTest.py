import numpy as np
import cv2
import datetime

from RoadImage import RoadImage
from lanes.image_debugtools import draw_analysed_image

'''
    Class to test the lane detection algorithm
    using one of the images from our hard_drive
'''

if __name__ == "__main__":

    # Load a color image in grayscale
    img = cv2.imread('./images/roi_test.png', -1)
    
    # Create a new instance of the class that
    # analyses road images
    ri = RoadImage(img)
    
    # Results from the analysis
    points = ri.analyse()
    raw_image = ri.get_image()
    
    draw_analysed_image(raw_image, points)

    print("Distance: {}".format(ri.center_offset()))

    # Show the results
    cv2.imshow("Result", raw_image)
    cv2.imshow("Hough Output", ri.get_hough())

    cv2.waitKey(0)
    cv2.destroyAllWindows()