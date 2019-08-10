import cv2

'''
    Some helpful functions for 
    debugging purposes.
'''

LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_INFO = "INFO"

def log(msg, level=LOG_LEVEL_DEBUG):
    
    CSTART = '\33[35m'
    CEND = '\33[0m'

    if level is LOG_LEVEL_ERROR:
        CSTART = '\33[41m'
    elif level is LOG_LEVEL_INFO:
        CSTART = '\33[34m'
    elif level is LOG_LEVEL_WARNING:
        CSTART = '\33[33m'
    
    print(CSTART + level + ": " + msg + CEND)

def draw_analysed_image(image, analysed_points):
    
    if analysed_points[0] is not None:
        left_bottom = tuple([int(x) for x in analysed_points[0]])
        cv2.circle(image, left_bottom, 5, (255,0,0), 5)  

    if analysed_points[1] is not None:
        intersection = tuple([int(x) for x in analysed_points[1]])
        cv2.circle(image, intersection, 5, (255,0,0), 5)  
        cv2.line(image, left_bottom, intersection, (0, 255, 0), 3)
    
    if analysed_points[2] is not None:
        right_bottom = tuple([int(x) for x in analysed_points[2]])
        cv2.circle(image, right_bottom, 5, (255,0,0), 5)  
        cv2.line(image, right_bottom, intersection, (0, 255, 0), 3)
    
    if analysed_points[3] is not None:
        bisection = tuple([int(x) for x in analysed_points[3]])
        cv2.circle(image, bisection, 5, (255,0,255), 5)  

    if analysed_points[4] is not None:
        center = tuple([int(x) for x in analysed_points[4]])
        cv2.circle(image, center, 5, (255,255,0), 5) 