from typing import Tuple

import numpy as np
import cv2
import rospy


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for reactive control
                            using the masked left lane markings (numpy.ndarray)
    """

    steer_matrix_left_weight = rospy.get_param(f"/steer_matrix_left_weight")

    # row_values = -0.03 * np.linspace(0.2, 1, shape[1])
    # steer_matrix_left = np.tile(row_values, (shape[0], 1))  # make it 2d

    # steer_matrix_left = -0.002 * np.outer(np.linspace(0.3, 1, shape[0]), np.linspace(0.3, 1, shape[1])) ** 2

    steer_matrix_left = steer_matrix_left_weight * np.ones(shape)

    # steer_matrix_left[:shape[0]//3,:] = 0  # removing impact from horizon like pannels on top of the duckiematrix loop etc

    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for reactive control
                             using the masked right lane markings (numpy.ndarray)
    """

    steer_matrix_right_weight = rospy.get_param(f"/steer_matrix_right_weight")

    # row_values = 0.02 * np.linspace(0.2, 1, shape[1])[::-1]
    # steer_matrix_right = np.tile(row_values, (shape[0], 1))  # make it 2d

    # steer_matrix_right = 0.001 * np.outer(np.linspace(0.3, 1, shape[0]), np.linspace(1, 0.3, shape[1])) ** 2

    steer_matrix_right = steer_matrix_right_weight * np.ones(shape)

    # steer_matrix_right[:shape[0]//3,:] = 0  # removing impact from horizon like pannels on top of the duckiematrix loop etc

    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape

    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sigma = 6
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    threshold = 5
    mask_mag = (Gmag > threshold)

    white_lower_hsv = np.array([0, 0, 150])
    white_upper_hsv = np.array([179, 50, 255])
    yellow_lower_hsv = np.array([20, 75, 75])
    yellow_upper_hsv = np.array([30, 255, 255])

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(w/2)):w + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(w/2))] = 0

    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    mask_left_edge = mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    return mask_left_edge, mask_right_edge


if __name__ == "__main__":
    print(get_steer_matrix_left_lane_markings((5,5)))
    print(get_steer_matrix_right_lane_markings((5,5)))
