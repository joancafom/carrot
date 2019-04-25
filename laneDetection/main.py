from lane_marking import LaneMarking
import cv2

for c in range(1, 9):
    image = cv2.imread("./base_imagenes/bd_{}.jpg".format(c))
    # (h, w, d) = image.shape
    # roi = image[h//2:h-20, 0:w]

    clase = LaneMarking(image)
    clase.preprocess_image()
    clase.compute_lanes()
    clase.dissect_image()
    cv2.imshow("Marry the Night", clase.get_dissected_image())
    cv2.waitKey(0)