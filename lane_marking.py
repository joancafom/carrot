import cv2
import numpy as np

"""
Detects both lane markings on an image and computes
the median or bisection.

Both lanes must be visible in the image.
"""

class LaneMarking:

    def __init__(self, input_image):

        # Image to be analysed
        self.input_image = input_image

        # Intermediate Images
        self.grayscale_image = None
        self.binarized_image = None
        self.edged_image = None
        self.hough_image = None
        self.lined_image = None
        self.dissected_image = None

        # Shared properties
        (self.h, self.w, _) = self.input_image.shape
        self.bottom_center = (self.w/2, self.h)
        self.hough_lines = None

        # Left Lane Properties:
        # Slope, Independent Term, Drawing Points
        self.left_m = None
        self.left_n = None
        self.top_left_point = None
        self.bottom_left_point = None

        # Left Lane Properties:
        # Slope, Independent Term, Drawing Points
        self.right_m = None
        self.right_n = None
        self.top_right_point = None
        self.bottom_right_point = None

    def get_input_image(self):
        return self.input_image

    def get_grayscale_image(self):
        return self.grayscale_image
    
    def get_binarized_image(self):
        return self.binarized_image
    
    def get_edged_image(self):
        return self.edged_image
    
    def get_hough_image(self):
        return self.hough_image
    
    def get_lined_image(self):
        return self.lined_image
    
    def get_dissected_image(self):
        return self.dissected_image
    
    '''
        Pipeline that obtains an image that's ready to
        be applied our algorithm for lane detection.
    '''
    def preprocess_image(self, binarize_threshold=120, binarize_maxval=255, canny_lower_threshold=50, canny_upper_threshold=150,
    hough_rho_resolution=1, hough_thetha_resolution=np.pi/100, hough_threshold_votes=70, hough_minLineLength=5, 
    hough_maxLineGap=10, line_color=(150,20,150)):
        assert self.input_image is not None

        self.grayscale_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
        self.binarized_image = cv2.threshold(self.grayscale_image, binarize_threshold, binarize_maxval, cv2.THRESH_BINARY)[1]
        self.edged_image = cv2.Canny(self.binarized_image, canny_lower_threshold, canny_upper_threshold) 
        
        # Create a copy in which to draw detected lines
        self.hough_image = self.input_image.copy()
        self.hough_lines = cv2.HoughLinesP(self.edged_image, hough_rho_resolution, hough_thetha_resolution, 
            hough_threshold_votes, minLineLength=hough_minLineLength, maxLineGap=hough_maxLineGap)
        
        # Draw lines found by Hough
        for line in self.hough_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(self.hough_image, (x1, y1), (x2, y2), line_color, 3)
        
        print("number of lines: ", len(self.hough_lines))

    '''
        Computes and draws the lanes given a image to which
        hough transform was correctly applied.
    '''
    def compute_lanes(self):
        assert self.hough_lines is not None

        self.lined_image = self.hough_image.copy()
        ((left_m, left_n), (right_m, right_n)) = self.compute_lanes_parameters()
        
        self.left_m = left_m
        self.left_n = left_n
        self.right_m = right_m
        self.right_n = right_n
        
        # Left Lane Line's Points: The upper-most and bottom-most points corresponding 
        # to the line, in order to draw it
        self.top_left_point, self.bottom_left_point = self.compute_line_drawing_points(self.left_m, self.left_n)

        # Right Lane Line's Points: The upper-most and bottom-most points corresponding 
        # to the line, in order to draw it
        self.top_right_point, self.bottom_right_point = self.compute_line_drawing_points(self.right_m, self.right_n)

        cv2.circle(self.lined_image, self.top_left_point, 3, (0, 255, 255), 3)
        cv2.circle(self.lined_image, self.bottom_left_point, 3, (0, 255, 255), 3)
        cv2.line(self.lined_image, self.top_left_point, self.bottom_left_point, (0, 0, 255), 3)
        cv2.circle(self.lined_image, self.top_right_point, 3, (0, 255, 255), 3)
        cv2.circle(self.lined_image, self.bottom_right_point, 3, (0, 255, 255), 3)
        cv2.line(self.lined_image, self.top_right_point, self.bottom_right_point, (0, 0, 255), 3)

    '''
        Computes the median/bisection of two secant lines

        Returns the distance between the center of the image and the bottom-most point
        of the line that passes through both the intercenter and the baricenter.

        Return: [distance_intercenter, distance_baricenter]
    '''
    def dissect_image(self, point_color=(255, 0, 0), line_color=(100, 255, 100)):
        assert self.left_m is not None and self.left_n is not None and \
            self.right_m is not None and self.right_n is not None

        # First of all, we need the intersection point between the two lines
        # We can do so, by solving a simple equation system with two unknowns:
        #----------------
        # | y = mx + n
        # | y = m'x + n' 
        #----------------
        # x = (n - n') / (m' - m)
        intersection_x = (self.left_n - self.right_n) / (self.right_m - self.left_m)
        intersection_y = self.left_m * intersection_x + self.left_n
        intersection = (intersection_x, intersection_y)

        self.dissected_image = self.lined_image.copy()

        cv2.circle(self.dissected_image, self.int_point(intersection), 3, point_color, 3)
        cv2.line(self.dissected_image, self.int_point(intersection), self.int_point(self.bottom_center), line_color, 3)

        # ----- INTERCENTER ----
        # We compute the intercenter of a triangle formed by the intersection point and two 
        # arbitrary points of each line
        # Color: DARK PINK
        incenter = self.compute_incenter(intersection, self.bottom_left_point, self.bottom_right_point)

        cv2.circle(self.dissected_image, self.int_point(incenter), 3, point_color, 3)

        m = self.compute_slope(incenter[0], incenter[1], intersection[0], intersection[1])
        n = self.compute_independent(incenter[0], incenter[1], intersection[0], intersection[1])
        top_point_i, bottom_point_i = self.compute_line_drawing_points(m, n)
        cv2.line(self.dissected_image, top_point_i, bottom_point_i, (144, 66, 244), 3)

        # ----- BARICENTER ----
        # We compute the baricenter of a triangle form by the intersection point and two 
        # arbitrary points of each line
        # Color: LIGHT PINK
        baricenter = self.compute_baricenter(intersection, self.bottom_left_point, self.bottom_right_point)

        cv2.circle(self.dissected_image, self.int_point(baricenter), 3, point_color, 3)

        m = self.compute_slope(baricenter[0], baricenter[1], intersection[0], intersection[1])
        n = self.compute_independent(baricenter[0], baricenter[1], intersection[0], intersection[1])
        top_point_b, bottom_point_b = self.compute_line_drawing_points(m, n)
        cv2.line(self.dissected_image, top_point_b, bottom_point_b, (144, 130, 244), 3)

        # ----- Distance to INTERCENTER -----
        distance_i = self.compute_distance_points(self.bottom_center, bottom_point_i)
        distance_b = self.compute_distance_points(self.bottom_center, bottom_point_b)
        cv2.putText(self.dissected_image, "DISTANCE BISECTION: {}px".format(distance_i), (10, 35), cv2.FONT_HERSHEY_COMPLEX, 1.5, (144, 66, 244),3, lineType=cv2.LINE_AA)
        cv2.putText(self.dissected_image, "DISTANCE MEDIAN: {}px".format(distance_b), (10, 85), cv2.FONT_HERSHEY_COMPLEX, 1.5, (144, 130, 244), 3, lineType=cv2.LINE_AA)

        return (distance_i, distance_b)

    # Auxiliary Methods

    '''
        Computes the parameters that characterize the lane lines,
        that is, their slope and independent terms.

        Given the expresion:

                        'y = mx + n'

        'm' is defined as the slope of the line
        'n' is defined as the independent term of the line

        Returns: a tuple of the form ((m_1,n_1), (m_2,n_2))
    '''
    def compute_lanes_parameters(self):

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
        for line in self.hough_lines:
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

        # Draw the selected points
        cv2.circle(self.lined_image, left_point, 3, (0, 255, 0), 3)
        cv2.circle(self.lined_image, right_point, 3, (0, 255, 0), 3)
        cv2.circle(self.lined_image, left_point_partner, 3, (0, 0, 255), 3)
        cv2.circle(self.lined_image, right_point_partner, 3, (0, 0, 255), 3)

        # Compute the parameters that characterize both lines (the slope and the independent term)
        left_m = self.compute_slope(left_point[0], left_point[1], left_point_partner[0], left_point_partner[1])
        right_m = self.compute_slope(right_point[0], right_point[1], right_point_partner[0], right_point_partner[1])

        left_n = self.compute_independent(left_point[0], left_point[1], left_point_partner[0], left_point_partner[1])
        right_n = self.compute_independent(right_point[0], right_point[1], right_point_partner[0], right_point_partner[1])
        return((left_m, left_n), (right_m, right_n))
    
    '''
        Computes the slope of two given points.
        
        If the slope function is not defined (that is,
        when x2 = x1), we substract one pixel from
        either x2 or x1 to keep it bounded.
    '''
    def compute_slope(self, x1, y1, x2, y2):
        print("x1, y1, x2, y2: {}, {}, {}, {}".format(x1, y1, x2, y2))
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

    def compute_baricenter(self, a, b, c):

        x_i = (a[0] + b[0] + c[0]) / 3
        y_i = (a[1] + b[1] + c[1]) / 3

        return (x_i, y_i)

    def int_point(self, point):
        return (int(point[0]), int(point[1]))

    '''
        Computes the distance between two points
    '''
    def compute_distance_points(self, pointA, pointB):
        res = None

        if pointA and pointB:
            vector = [pointB[0] - pointA[0], pointB[1] - pointA[1]]
            res = np.linalg.norm(vector)


        return res
