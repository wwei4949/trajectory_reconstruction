from ReadCameraModel import ReadCameraModel
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Extract the camera parameters and compute the camera’s intrinsic matrix K
fx, fy, cx, cy, _, LUT = ReadCameraModel('./Oxford_dataset_reduced/model')

K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

print("Intrinsic matrix: \n", K)

# sort all image filenames
image_filenames = sorted(os.listdir('./Oxford_dataset_reduced/images'), key=lambda x: int(os.path.splitext(x)[0]))

# initialize SIFT detector
sift = cv2.SIFT_create()


# initialize the FLANN matcher
# SOURCE: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# initialize lists to store images, keypoints, and other data for visualization
images = []
keypoints = []
keypoint_images = []
matched_images = []
fundamental_matrices = []
point_correspondences = []
gray_images = []
trajectory = []
pts = []
prev_image = None
prev_kp = None
prev_des = None

# initialize the rotation and translation matrices for the first image
rotation = np.eye(3)
translation = np.zeros((3, 1))

for i, filename in enumerate(image_filenames):
    # load the image
    img = cv2.imread(os.path.join('./Oxford_dataset_reduced/images', filename), flags=-1)

    # demosaic the image
    color_image = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)

    # convert the image to grayscale for SIFT
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # find keypoints and descriptors
    kp, des = sift.detectAndCompute(gray_image, None)

    # add the image and keypoints to the lists
    images.append(color_image)
    gray_images.append(gray_image)
    keypoints.append(kp)

    # draw keypoints on the first, 188th, and last images
    if i in {0, 187, len(image_filenames) - 1}:
        keypoint_image = cv2.drawKeypoints(color_image, kp, None)
        keypoint_images.append(keypoint_image)

    # if not the first image
    if i > 0:
        # match the descriptors from the previous image to the current image
        matches = flann.knnMatch(prev_des, des, k = 2)
        good_matches = []

        # apply Lowe's ratio test to the matches to get rid of bad matches
        # SOURCE: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        matches = good_matches

        # draw the matches on the first and second images as well as the last and second-to-last images
        if i in {1, len(image_filenames) - 1}:
            matched_image = cv2.drawMatches(prev_color_image, prev_kp, color_image, kp, matches, None)
            matched_images.append(matched_image)


        # good matches keypoints
        src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches])

        # Using the matched keypoints you just identified, estimate the fundamental matrix between the two frames
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

        # keep the fundamental matrix and the point correspondences for the first image for visualization of epilines
        if i == 1:
            fundamental_matrices.append(F)
            point_correspondences.append((src_pts, dst_pts))
            pts.append(src_pts[mask.ravel() == 1])
            pts.append(dst_pts[mask.ravel() == 1])


        # inlier points
        src_pts = src_pts[mask.ravel() == 1]
        dst_pts = dst_pts[mask.ravel() == 1]

        # Estimate the Essential Matrix E from the Fundamental Matrix F by accounting for the calibration parameters
        E = np.dot(K.T, np.dot(F, K))

        # Decompose E into a physically realizable translation T and rotation R
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, focal=fx, pp=(cx, cy))

        # get camera motion from the essential matrix
        R = R.transpose()
        t = -np.dot(R, t)

        rotation = np.dot(prev_rot, R)
        translation = np.dot(prev_rot, t) + prev_trans

        trajectory.append(translation)

    # store the current image, keypoints, and descriptors for the next iteration
    prev_image = gray_image
    prev_color_image = color_image
    prev_kp = kp
    prev_des = des
    prev_rot = rotation
    prev_trans = translation


######### OUTPUTS ########
# output the first image, the 188th image, and the last image after demoasicing
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(images[0])
plt.title('First image')

plt.subplot(1, 3, 2)
plt.imshow(images[187])
plt.title('188th image')

plt.subplot(1, 3, 3)
plt.imshow(images[-1])
plt.title('Last image')

plt.show()

# draw the keypoints on the first, 188th, and last images

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(keypoint_images[0], cv2.COLOR_BGR2RGB))
plt.title('First image')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(keypoint_images[1], cv2.COLOR_BGR2RGB))
plt.title('188th image')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(keypoint_images[2], cv2.COLOR_BGR2RGB))
plt.title('Last image')

plt.show()

# draw the matches between the first and second images, and between the second to last and last images
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(matched_images[0], cv2.COLOR_BGR2RGB))
plt.title('Matches between first and second frames')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(matched_images[1], cv2.COLOR_BGR2RGB))
plt.title('Matches between second to last and last frames')

plt.show()

# function to draw epilines on the images
# SOURCE: https://docs.opencv.org/4.5.2/da/de9/tutorial_py_epipolar_geometry.html
def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1_copy = cv2.cvtColor(img1.copy(), cv2.COLOR_GRAY2BGR)
    img2_copy = cv2.cvtColor(img2.copy(), cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1_copy = cv2.line(img1_copy, (x0, y0), (x1, y1), color, 1)
        img1_copy = cv2.circle(img1_copy, tuple(map(int, pt1)), 5, color, -1)  # Convert pt1 to integers
        img2_copy = cv2.circle(img2_copy, tuple(map(int, pt2)), 5, color, -1)  # Convert pt2 to integers
    return img1_copy, img2_copy

# compute the fundamental matrix between the first and second frames
F = fundamental_matrices[0]

# extract the point correspondences between the first and second frames
points_previous, points_current = point_correspondences[0]

# compute the epilines in the second image corresponding to the points in the first image and draw them on the first image
lines1 = cv2.computeCorrespondEpilines(points_current.reshape(-1,1,2), 2, F).reshape(-1,3)
img5, img6 = drawlines(gray_images[0], gray_images[1], lines1, points_previous, points_current)
lines2 = cv2.computeCorrespondEpilines(points_previous.reshape(-1,1,2), 1, F).reshape(-1,3)
img3, img4 = drawlines(gray_images[1], gray_images[0], lines2, points_current, points_previous)

plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)
plt.imshow(img5)
plt.title('First image with epilines')

plt.subplot(1, 2, 2)
plt.imshow(img3)
plt.title('Second image with epilines')

plt.show()

# convert the list of translations to a 3D NumPy array
trajectory = np.array(trajectory).reshape(-1, 3)

# plot 2d trajectory
plt.figure()
plt.plot(trajectory[:, 0], trajectory[:, 2])
plt.title("Camera Trajectory")
plt.xlabel("X")
plt.ylabel("Z")

# standardize scale of axes
plt.axis('equal')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(trajectory[:,0], trajectory[:,2], trajectory[:,1])

ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')

# standardize scale of axes
range = np.array([trajectory[:, 0].max() - trajectory[:, 0].min(), trajectory[:, 2].max() - trajectory[:, 2].min(), trajectory[:, 1].max() - trajectory[:, 1].min()]).max() / 2.0

x = (trajectory[:, 0].max() + trajectory[:, 0].min()) * 0.5
y = (trajectory[:, 2].max() + trajectory[:, 2].min()) * 0.5
z = (trajectory[:, 1].max() + trajectory[:, 1].min()) * 0.5
ax.set_xlim(x - range, x + range)
ax.set_ylim(y - range, y + range)
ax.set_zlim(z - range, z + range)

plt.show()
