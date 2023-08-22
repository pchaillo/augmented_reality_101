#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Augmented Reality with Python 3 and OpenCV 4.2
"""

# __author__ = "ma. fernanda rodriguez r."
# __email__ = "mafda13@gmail.com"
# __created__ = "Thu 14 May 2020 11:40:54 -0300"
# __modified__ = "Thu 29 May 2020 15:13:00 -0300"


import cv2
import math
import threading
import numpy as np
import matplotlib.pyplot as plt
from objloader_simple import *
from collections import deque
import pyrealsense2 as rs
import src.project_and_display as proj


class RealSenseCapture:
    """RealSense Camera Capture
    """

    def __init__(self, res=(640, 480)):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, res[0], res[1], rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.q = deque()
        self.status = "init"

        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

        while self.status == "init":
            pass

        assert self.status == "capture", "Failed to open RealSense camera"

    def _reader(self):
        """Read frames from the RealSense camera, keeping only the most recent one
        """

        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("[error] no color frame")
                break

            frame = np.asanyarray(color_frame.get_data())
            self.q.append(frame)

            self.status = "capture"

            while len(self.q) > 1:
                self.q.popleft()

        self.status = "failed"

        

    def read(self):
        return self.q[-1]

    def release(self):
        self.pipeline.stop()


def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]

    # Normalize vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    # Compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(
        c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
    )
    rot_2 = np.dot(
        c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
    )
    rot_3 = np.cross(rot_1, rot_2)

    print("Rotation1 :")
    print(rot_1)

    # Compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T

    # return np.dot(camera_parameters, projection) # version du tutoriel
    return (camera_parameters @ projection) # version tinhinane


def render(frame, obj, projection, referenceImage, scale3d, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale3d
    h, w = referenceImage.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)

        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        framePts = np.int32(dst)

        cv2.fillConvexPoly(frame, framePts, (137, 27, 211))

    return frame


def main():

    # ============== Read data ==============

    # Load 3D model from OBJ file
    obj = OBJ("./models/chair.obj", swapyz=True)

    # Scale 3D model
    scale3d = 8

    # Matrix of camera parameters
    # camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]]) # valeur du tutoriel
    camera_parameters = np.array([ # matrice intrinsèque de la camera realsense
    [382.43677, 0,319.68753],
    [0, 382.43677, 240.88152],
    [0, 0, 1] ] )

    # Minimum number of matches
    MIN_MATCHES = 15

    # ============== Reference Image ==============

    # Load reference image and convert it to gray scale
    referenceImage = cv2.imread("./img/referenceImage.jpg", 0)

    # ================== Recognize ================

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # create brute force  matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Compute model keypoints and its descriptors
    referenceImagePts, referenceImageDsc = orb.detectAndCompute(referenceImage, None)

    # =============== Source Images ==============

    # Init RealSense camera capture
    cap = RealSenseCapture()

    while True:
        # read the current frame
        frame = cap.read()

        # ============== Recognize =============

        # Compute scene keypoints and its descriptors
        sourceImagePts, sourceImageDsc = orb.detectAndCompute(frame, None)

        # ============== Matching =============

        # Match frame descriptors with model descriptors
        matches = bf.match(referenceImageDsc, sourceImageDsc)

        # Sort them in the order of their distance
        matches = sorted(matches, key=lambda x: x.distance)

        # ============== Homography =============

        # Apply the homography transformation if we have enough good matches
        if len(matches) > MIN_MATCHES:
            # Get the good key points positions
            sourcePoints = np.float32(
                [referenceImagePts[m.queryIdx].pt for m in matches]
            ).reshape(-1, 1, 2)
            destinationPoints = np.float32(
                [sourceImagePts[m.trainIdx].pt for m in matches]
            ).reshape(-1, 1, 2)

            # Obtain the homography matrix
            homography, _ = cv2.findHomography(
                sourcePoints, destinationPoints, cv2.RANSAC, 5.0
            )

            # Apply the perspective transformation to the source image corners
            h, w = referenceImage.shape
            corners = np.float32(
                [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
            ).reshape(-1, 1, 2)
            transformedCorners = cv2.perspectiveTransform(corners, homography)

            # Draw a polygon on the second image joining the transformed corners
            frame = cv2.polylines(
                frame, [np.int32(transformedCorners)], True, 255, 3, cv2.LINE_AA,
            )

            # ================= Pose Estimation ================

            # obtain 3D projection matrix from homography matrix and camera parameters
            projection = projection_matrix(camera_parameters, homography) # version du tutoriel
            # projection = camera_parameters @ homography # version Tinhinane

            # print(projection)


            # project cube or model
            # frame = render(frame, obj, projection, referenceImage, scale3d, False) # version du tutoriel
            h = 480
            w = 640
            # frame = proj.project_and_display_without_colors(frame, obj, projection,h,w)
            colors = [0,1,0]
            frame = proj.project_and_display(frame, obj, projection,colors)

            # ===================== Display ====================

            # show result
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            print("Not enough matches are found - %d/%d" % (len(matches), MIN_MATCHES))

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()
