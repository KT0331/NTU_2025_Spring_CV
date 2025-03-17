import os
import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=5.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--plot_mode', default='v', help='v means visualize DoG, f means plot feature points')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float32)
    img_rgb = cv2.imread(args.image_path)

    ### TODO ###
    DoG = Difference_of_Gaussian(args.threshold)
    os.makedirs('./output_image', exist_ok=True)
    
    if args.plot_mode == 'v':
        # find keypoint from DoG and sort it
        _, dog_images = DoG.get_keypoints(img)  
        for i in range(len(dog_images)):
            if i < 4:
                cv2.imwrite('./output_image/DOG1-{}.png'.format(i+1), dog_images[i].astype(np.uint8))
            else:
                cv2.imwrite('./output_image/DOG2-{}.png'.format(i-3), dog_images[i].astype(np.uint8))

    elif args.plot_mode == 'f':
        keypoints, _ = DoG.get_keypoints(img)  

        for i in range(len(keypoints)):
            cv2.circle(img_rgb, (keypoints[i][1], keypoints[i][0]), 5, (0, 0, 255), 0)

        cv2.imwrite('./output_image/DOG_f_{}.png'.format(args.threshold), img_rgb)
    else:
        print("Invalid Plot Type!")


if __name__ == '__main__':
    main()