import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)
np.random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(im1,None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        img1_point=[]
        img2_point=[]
        N = len(matches)
        for match in matches:
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            img1_point.append(keypoints1[img1_idx].pt)
            img2_point.append(keypoints2[img2_idx].pt)

        img1_point = np.array(img1_point) #(N, 2)
        img2_point = np.array(img2_point) #(N, 2)
        img1_point_homogeneous = np.concatenate((img1_point, np.ones((N, 1))), axis=1) #(N, 3)
        img2_point_homogeneous = np.concatenate((img2_point, np.ones((N, 1))), axis=1) #(N, 3)
        
        # TODO: 2. apply RANSAC to choose best H
        ransac_iter = 5000
        inlier_threshold = 10
        best_inlier_num = 0
        for i in range(ransac_iter):
            # 8 unknown value need 4 match point
            selected_idx = np.random.choice(N, 4) # Return 4 random numbers (range [0, N-1])
            img1_point_selected = img1_point[selected_idx] #(4, 2)
            img2_point_selected = img2_point[selected_idx] #(4, 2)
            H = solve_homography(img2_point_selected, img1_point_selected)

            pred_img1_point = H @ img2_point_homogeneous.T #(3, N)
            pred_img1_point = (pred_img1_point / pred_img1_point[2]).T #(N, 3)
            error = np.linalg.norm(pred_img1_point - img1_point_homogeneous, axis=1) #(N,)
            n_inlier = np.sum((error < inlier_threshold).astype(int))
            
            if n_inlier > best_inlier_num:
                best_inlier_num = n_inlier
                best_H = H

        # TODO: 3. chain the homographies
        last_best_H = last_best_H @ best_H

        # TODO: 4. apply warping
        out = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')

    return out 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)