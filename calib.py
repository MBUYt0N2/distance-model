import cv2
import numpy as np
import glob

# Define the chessboard size
chessboard_size = (9, 6)

# Define arrays to store object points and image points from all images
obj_points = []  # 3D point in real-world space
img_points = []  # 2D points in image plane

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)

# Load images
images = glob.glob("calibration_images/*.jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points and image points (after refining them)
    if ret:
        obj_points.append(objp)
        img_points.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow("img", img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration to find intrinsic and extrinsic parameters
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

# K is the intrinsic matrix
print("Intrinsic Matrix (K):\n", K)

# Compute the extrinsic matrix for the first image
rvec = rvecs[0]
tvec = tvecs[0]
R, _ = cv2.Rodrigues(rvec)
P = np.hstack((R, tvec.reshape(-1, 1)))
print("Extrinsic Matrix (P) for the first image:\n", P)
print(rvec)
print(tvec)
np.save("calibration_params", {"K": K, "dist": dist, "R":rvec, "t":tvec, "P":P})
