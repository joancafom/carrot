import cv2
import numpy as np

from debugtools import log, LOG_LEVEL_ERROR, LOG_LEVEL_WARNING

"""
Class that process & analyses a road image in 
order to identify the lanes that appear in it
and to compute the offset distance that separates
the driver to the center of the lane.

The image of the road can represent any of the
following scenarios, so the class is well 
prepared to fully support them:

    * Image with two lanes [RI-A]
    * Image with one one lane (either left or right) [RI-B]
    * An image without any lanes [RI-C]


The following assumptions on the image are made:

    * Lane marks are black
    * There are no other black marks in the image
    * The environment is light compared to the lanes,
    it does not need to be regular or unicolored.
    * There is enough contrast between the lane marks
    and the environment

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

    ###### GETTERS #####

    # Cascade Getters that allow access to
    # every single image in our pipeline

    def get_image(self):
        # Prevent modifications in the view by making a copy
        return self.image.copy()

    def get_grayscale(self):
        return cv2.cvtColor(self.get_image(), cv2.COLOR_BGR2GRAY)
    
    def get_clahe(self):
        return cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(self.get_grayscale())
    
    def get_gamma(self):
        return cv2.LUT(self.get_clahe(), np.array([((i / 255.0) ** 0.5) * 255 for i in np.arange(0, 256)]).astype("uint8"))
    
    def get_binarized(self, binarize_threshold=120, binarize_maxval=255):
        return cv2.threshold(self.get_gamma(), binarize_threshold, binarize_maxval, cv2.THRESH_BINARY)[1]
    
    def get_edged(self, canny_lower_threshold=50, canny_upper_threshold=150):
        return cv2.Canny(self.get_binarized(), canny_lower_threshold, canny_upper_threshold)

    # This method is just for debugging purposes
    '''
    Returns an instance of the raw image in which lines 
    resulting from the application of the Hough Transform
    have been drawn
    '''
    def get_hough(self, hough_rho_resolution=1, hough_thetha_resolution=np.pi/100, hough_threshold_votes=30, hough_minLineLength=5, 
    hough_maxLineGap=10):
        hough_image = self.get_image().copy()

        hough_lines = cv2.HoughLinesP(self.get_edged(), hough_rho_resolution, hough_thetha_resolution, 
        hough_threshold_votes, minLineLength=hough_minLineLength, maxLineGap=hough_maxLineGap)

        if hough_lines is not None:
            # Draw lines found by Hough
            for line in hough_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(hough_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        else:
            log("[RI-C] No candidate Hough Lines were found", level=LOG_LEVEL_ERROR)

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
    def analyse(self, hough_rho_resolution=1, hough_thetha_resolution=np.pi/100, hough_threshold_votes=30, hough_minLineLength=5, 
    hough_maxLineGap=10):

        log("Watch out for image size! Less pixels means Less votes")

        # ---------- STEP 1: Extract Hough lines ----------
        #  
        ## In this step we identify the possible lines that appear in our image.
        ## In order to be considered as such, the algorithm goes through a voting process
        ## and each of the candidates must get enough votes to be considered a line.

        hough_lines = cv2.HoughLinesP(self.get_edged(), hough_rho_resolution, hough_thetha_resolution, 
            hough_threshold_votes, minLineLength=hough_minLineLength, maxLineGap=hough_maxLineGap)

        ## If no candidate lines were identified, we are in [RI-C] and we cannot successfully 
        ## estimate the distance, we don't have enough reference.
        if hough_lines is None:
            
            log("[RI-C] No candidate Hough Lines were found", level=LOG_LEVEL_ERROR)
            return (None, None, None, None, None)
        
        ##
        #
        # -------------------------------------------------

        # ---------- STEP 2: Figure out where lanes are ----------
        #  
        ## We now go through every candidate line and keep only the two of them which
        ## correspond to the lane markings (one for each side)

        ((left_point, left_point_partner), (right_point, right_point_partner)) = self._compute_nearest_lanes(hough_lines)

        ## The two lane markings must be separated at least 100px of each other.
        ## If not, we consider that one of the markings is wrong and drop one of them
        ## in order to apply the reflection method (to estimate the distance) 
        lanes_dist = self.compute_distance_points(left_point, right_point)
        lanes_dist2 = self.compute_distance_points(left_point, right_point_partner)

        if lanes_dist and lanes_dist < MIN_INTERLANE_PX or lanes_dist2 and lanes_dist2 < MIN_INTERLANE_PX:
            right_point = None
            right_point_partner = None

        ##
        #
        # -------------------------------------------------

        # ---------- STEP 3: Extend lanes lines so they take up the whole image ----------
        #
        ## We need to extend the lines in order to ensure they pass by the x axis,
        ## which is the bottom-most point and thus it is less altered by perspective
        ## so that we can correctly compute the distance.
        ##
        ## Up to this point, we need to figure out what lanes where identified correctly
        ## (left, right or both) (we have cleared out the possibility of both of them being
        ## incorrectly identified in the previous assertion)

        if left_point is not None:

            # Left lane was correctly identified: Either [RI-A] or [RI-B]

            # Get the parameters the characterize left lane mark before we proceed
            left_m = self.compute_slope(left_point[0], left_point[1], left_point_partner[0], left_point_partner[1])
            left_n = self.compute_independent(left_point[0], left_point[1], left_point_partner[0], left_point_partner[1])

            if right_point is None:

                # We are in case [RI-B], only left lane was correctly identified
                log("[RI-B] - Only Left lane was correctly identified")

                # We reflect the left lane to obtain the right one and separate both 
                # lines in the picture by the width of a theoretical correct lane.
                right_n = self.compute_reflected_independent(left_m, left_point, INTERLANE_PX)
                right_m = -left_m
        
        if right_point is not None:

            # Right lane was correctly identified: Either [RI-A] or [RI-B]

            # Get the parameters the characterize right lane mark before we proceed
            right_m = self.compute_slope(right_point[0], right_point[1], right_point_partner[0], right_point_partner[1])
            right_n = self.compute_independent(right_point[0], right_point[1], right_point_partner[0], right_point_partner[1])

            if left_point is None:

                # We are in case [RI-B], only right lane was correctly identified
                log("[RI-B] - Only Right lane was correctly identified")

                # We reflect the right lane to obtain the left one and separate both 
                # lines in the picture by the width of a theoretical correct lane.
                left_n = self.compute_reflected_independent(right_m, right_point, INTERLANE_PX)
                left_m = -right_m

        ## Now, we have to ensure that both lines intersect. Two lines do not intersect
        ## if either they are parallel (same slope) or the same (same slope and independent)

        if left_m == right_m:

            if left_n == right_n:

                # Same lines, infinite intersection points, so we just reflect one
                # of them as if we had only identified that one.

                left_n = self.compute_reflected_independent(right_m, right_point, INTERLANE_PX)
                left_m = -right_m
            
            else:

                # Parallel lines: We change a the slope of one of them just a little
                # bit to transform it into intersecting

                left_m += 0.01

        ## We now obtain the drawing points (upper-most and bottom-most) of both lanes

        # Left Lane Line's Points: The upper-most and bottom-most points corresponding 
        # to the line, in order to draw it
        self.left_lane_top, self.left_lane_bottom = self.compute_line_drawing_points(left_m, left_n)

        # Right Lane Line's Points: The upper-most and bottom-most points corresponding 
        # to the line, in order to draw it
        self.right_lane_top, self.right_lane_bottom = self.compute_line_drawing_points(right_m, right_n)

        ##
        #
        # -------------------------------------------------

        # ---------- STEP 4: Compute the intersection of both lines ---------- 
        #
        ## We need the intersection point between the two lines
        ## We can do so, by solving a simple equation system with two unknowns:
        ## ----------------
        ## | y = mx + n
        ## | y = m'x + n' 
        ## ----------------
        ## x = (n - n') / (m' - m)

        intersection_x = (left_n - right_n) / (right_m - left_m)
        intersection_y = left_m * intersection_x + left_n
        intersection = (intersection_x, intersection_y)

        ##
        #
        # -------------------------------------------------
        
        # ---------- STEP 5: Compute the position of the car using the bisection of the angle ----------
        #
        ## INTERCENTER
        # We compute the intercenter of a triangle formed by the intersection point and two 
        # arbitrary points of each line, and after that, we get the middle point of the lane.
        incenter = self.compute_incenter(intersection, self.left_lane_bottom, self.right_lane_bottom)

        m = self.compute_slope(incenter[0], incenter[1], intersection[0], intersection[1])
        n = self.compute_independent(incenter[0], incenter[1], intersection[0], intersection[1])

        _, self.bisection_bottom = self.compute_line_drawing_points(m, n)

        ##
        #
        # -------------------------------------------------

        # Points are returned from left lane in a clockwise direction 
        return (self.left_lane_bottom, intersection, self.right_lane_bottom, self.bisection_bottom, self.bottom_center)
    
    '''
    Computes the distance (in px) between the car & the center
    of the lane
    '''
    def center_offset(self):
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

        # Don't need to check for hough_lines as this function
        # won't be called if it were invalid.
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
        
        # We must prevent the slope
        # from being 0 or inf

        if y1 == y2:
            
            # Slope that produces an almost
            # flat line without being zero
            return 0.001
        
        if x1 == x2:

            # Slope that produces an almost
            # straight line without being inf
            return 999
            
        return (y2-y1)/(x2-x1)

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

    '''
        Computes the independent coefficient of a line with slope the inverse
        to the original, separated to one point by a given distance.
        
        (x, y) & (x + dist, y)
        y = mx+n <-> n' = y + m'(x + dist)

    '''
    def compute_reflected_independent(self, original_slope, point, px_distance):
        return point[1] + original_slope*(point[0] + px_distance)
    
    def int_point(self, point):
        return (int(point[0]), int(point[1]))