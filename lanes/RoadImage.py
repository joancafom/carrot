import cv2
import numpy as np

"""
Class that process & analyses the raw image
received from the Android device wirelessly.

Both lanes must be visible in the image.
"""

class RoadImage:

    def __init__(self, input_image):

        assert input_image is not None

        # Image to be analysed
        self.image = input_image

        (self.h, self.w, _) = self.image.shape
        self.bottom_center = (self.w/2, self.h)

        self.left_lane_top = None
        self.right_lane_top = None
        self.left_lane_bottom = None
        self.right_lane_bottom = None

        self.bisection_bottom = None

        self.analyse()

    ###### GETTERS #####

    # Cascade Getters that allow access to
    # every single image in our pipeline

    def get_image(self):
        return self.image

    def get_grayscale(self):
        return cv2.cvtColor(self.get_image(), cv2.COLOR_BGR2GRAY)
    
    def get_binarized(self, binarize_threshold=120, binarize_maxval=255):
        return cv2.threshold(self.get_grayscale(), binarize_threshold, binarize_maxval, cv2.THRESH_BINARY)[1]
    
    def get_edged(self, canny_lower_threshold=50, canny_upper_threshold=150):
        return cv2.Canny(self.get_binarized(), canny_lower_threshold, canny_upper_threshold)

    # This method is just for debugging purposes
    '''
    Returns an instance of the raw image in which lines 
    resulting from the application of the Hough Transform
    have been drawn
    '''
    def get_hough(self, hough_rho_resolution=1, hough_thetha_resolution=np.pi/100, hough_threshold_votes=70, hough_minLineLength=5, 
    hough_maxLineGap=10):
        hough_image = self.get_image()

        hough_lines = cv2.HoughLinesP(self.get_edged(), hough_rho_resolution, hough_thetha_resolution, 
        hough_threshold_votes, minLineLength=hough_minLineLength, maxLineGap=hough_maxLineGap)

        # Draw lines found by Hough
        for line in hough_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

        return hough_image

    ##### MAIN FUNCTIONALITY #####

    '''
    Extracts the main relevant points of a raw image. Namely:

        - Left lane (Top & Bottom points)
        - Intersection point of both lanes
        - Rigth lane (Top & Bottom points)
        - Position of the car (center of the photo)
        - Midpoint of the lane
    
    Returns: (Bottom point of left lane, Intersection point, Bottom point of right lane,
    Position of the car (Bottom center of the photo), Midpoint of the lane)
    '''
    def analyse(self, hough_rho_resolution=1, hough_thetha_resolution=np.pi/100, hough_threshold_votes=70, hough_minLineLength=5, 
    hough_maxLineGap=10):

        print("Watch out for image size! Less pixels means less votes")

        # STEP 1: Extract Hough lines
        hough_lines = cv2.HoughLinesP(self.get_edged(), hough_rho_resolution, hough_thetha_resolution, 
            hough_threshold_votes, minLineLength=hough_minLineLength, maxLineGap=hough_maxLineGap)

        # STEP 2: Figure out where the lanes are 
        ((left_point, left_point_partner), (right_point, right_point_partner)) = self._compute_nearest_lanes(hough_lines)

        # STEP 3: Extend lanes lines so they take up the whole image

        # Compute the parameters that characterize both lines (the slope and the independent term)
        left_m = self.compute_slope(left_point[0], left_point[1], left_point_partner[0], left_point_partner[1])
        right_m = self.compute_slope(right_point[0], right_point[1], right_point_partner[0], right_point_partner[1])

        left_n = self.compute_independent(left_point[0], left_point[1], left_point_partner[0], left_point_partner[1])
        right_n = self.compute_independent(right_point[0], right_point[1], right_point_partner[0], right_point_partner[1])

        # Left Lane Line's Points: The upper-most and bottom-most points corresponding 
        # to the line, in order to draw it
        self.left_lane_top, self.left_lane_bottom = self.compute_line_drawing_points(left_m, left_n)

        # Right Lane Line's Points: The upper-most and bottom-most points corresponding 
        # to the line, in order to draw it
        self.right_lane_top, self.right_lane_bottom = self.compute_line_drawing_points(right_m, right_n)

        # STEP 4: Compute the intersection of both lines

        # We need the intersection point between the two lines
        # We can do so, by solving a simple equation system with two unknowns:
        #----------------
        # | y = mx + n
        # | y = m'x + n' 
        #----------------
        # x = (n - n') / (m' - m)
        intersection_x = (left_n - right_n) / (right_m - left_m)
        intersection_y = left_m * intersection_x + left_n
        intersection = (intersection_x, intersection_y)
        
        # STEP 5: Compute the position of the car using the bisection of the angle

        # ----- INTERCENTER ----
        # We compute the intercenter of a triangle formed by the intersection point and two 
        # arbitrary points of each line
        incenter = self.compute_incenter(intersection, self.left_lane_bottom, self.right_lane_bottom)

        m = self.compute_slope(incenter[0], incenter[1], intersection[0], intersection[1])
        n = self.compute_independent(incenter[0], incenter[1], intersection[0], intersection[1])

        _, self.bisection_bottom = self.compute_line_drawing_points(m, n)

        # Points are returned from left lane in a clockwise direction 
        return (self.left_lane_bottom, intersection, self.right_lane_bottom, self.bisection_bottom, self.bottom_center)
    
    '''
    Computes the distance (in px) between the car & the center
    of the lane
    '''
    def center_offset(self):
        assert self.bottom_center is not None
        assert self.bisection_bottom is not None

        return self.compute_distance_points(self.bottom_center, self.bisection_bottom)
    
    ###### AUXILIARY METHODS #####

    '''
    Obtains the two lines that are closest to the midpoint of the
    image (one on the left and another on the right). In order to do so,
    it iterates over the whole set of lines detected by Hough 
    (points) and calculates their distances to the midpoint. 
    '''
    def _compute_nearest_lanes(self, hough_lines):

        # Bottom-most left line, and its distante to the bottom-center
        left_point = None
        left_point_partner = None
        left_point_distance = None

        # Bottom-most right line, and its distante to the bottom-center
        right_point = None
        right_point_partner = None
        right_point_distance = None

        # Iterate through all points in the image to 
        # get the 2 bottom-most left and right points
        for line in hough_lines:
            x1, y1, x2, y2 = line[0]
        
            # The two points that compose the selected segment
            points_line = [(x1, y1), (x2, y2)]

            # Counter
            i = 0
            for point in points_line:
            
                i += 1
                # Vector that joins the point and the bottom center
                d_center_point = (point[0] - self.bottom_center[0], point[1] - self.bottom_center[1])
                distance = np.linalg.norm(d_center_point) # Distance = Modulus

                if self.bottom_center[0] >= point[0]:
                    # Point is on the left side
                    if left_point is None or left_point_distance > distance:
                        left_point = point
                        left_point_partner = points_line[(i%len(points_line))]
                        left_point_distance = distance
                else:
                    # Point is on the right side
                    if right_point is None or right_point_distance > distance:
                        right_point = point
                        right_point_partner = points_line[(i%len(points_line))]
                        right_point_distance = distance
        
        return ((left_point, left_point_partner), (right_point, right_point_partner))

    '''
        Computes the slope of two given points.
        
        If the slope function is not defined (that is,
        when x2 = x1), we substract one pixel from
        either x2 or x1 to keep it bounded.
    '''
    def compute_slope(self, x1, y1, x2, y2):
        return (y2-y1)/(x2-x1) if x2 != x1 else (y2-y1)

    '''
        Computes the independent term of a line that two given 
        points create.

        For a line expressed as 'y = mx + n', the independent term
        corresponds to 'n'.
        
        If the slope function is not defined (that is,
        when x2 = x1), we substract one pixel from
        either x2 or x1 to keep it bounded.
    '''
    def compute_independent(self, x1, y1, x2, y2):
        return (x1*(y1-y2))/(x2-x1) + y1 if x2 != x1 else x1*(y1-y2) + y1

    '''
        Given a line represented by its slope and independent term,
        computes the upper-most and bottom-most points in order
        to draw it
    '''
    def compute_line_drawing_points(self, m, n):

        # The upper-most point is in y = 0
        #   y = mx + n <--> x = y - n/m
        x = -n // m # y = 0
        top_point = (x.astype(int), 0) 

        # The bottom-most point is in y = height
        x = (self.h - n) // m # y = height
        bottom_point = (x.astype(int), self.h)
    
        return top_point, bottom_point

    '''
        Computes the incenter of a triangle given its three vertices
    '''
    def compute_incenter(self, a, b, c):

        d_ab = [b[0] - a[0], b[1] - a[1]]
        d_ab_module = np.linalg.norm(d_ab)
        d_ac = [c[0] - a[0], c[1] - a[1]]
        d_ac_module = np.linalg.norm(d_ac)
        d_bc = [b[0] - c[0], b[1] - c[1]]
        d_bc_module = np.linalg.norm(d_bc)

        sum_modules = d_ab_module + d_ac_module + d_bc_module

        x_i = (a[0]*d_bc_module + c[0]*d_ab_module + b[0]*d_ac_module) / sum_modules
        y_i = (a[1]*d_bc_module + c[1]*d_ab_module + b[1]*d_ac_module) / sum_modules

        return (x_i, y_i)

    '''
        Computes the distance between two points
    '''
    def compute_distance_points(self, pointA, pointB):
        res = None

        if pointA and pointB:
            vector = [pointB[0] - pointA[0], pointB[1] - pointA[1]]
            res = np.linalg.norm(vector)


        return res
    
    def int_point(self, point):
        return (int(point[0]), int(point[1]))