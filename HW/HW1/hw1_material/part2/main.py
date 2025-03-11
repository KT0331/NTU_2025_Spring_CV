import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    os.makedirs('./output_image', exist_ok=True)
    if (args.image_path == './testdata/1.png'):
        test_img = 1
    elif (args.image_path == './testdata/2.png'):
        test_img = 2
    else:
        print('Invalid input type')
    
    img_y = np.ones((img_rgb.shape[0], img_rgb.shape[1]))
    setting = []
    with open (args.setting_path) as f:
        for line in f:
            p = []
            p = line.split()
            for d in p:
                setting.append(d)

    sigma_s, sigma_r = setting[6].split(',')[1], setting[6].split(',')[3]
    sigma_s, sigma_r = int(sigma_s), float(sigma_r)
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)

    jbf_out_rgb = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)

    for i in range(1, 7):
        if i < 6:
            weight_r, weight_g, weight_b = setting[i].split(',')[0], setting[i].split(',')[1], setting[i].split(',')[2]
            weight_r, weight_g, weight_b = np.float64(weight_r), np.float64(weight_g), np.float64(weight_b)      
            img_y[:, :] = weight_r * img_rgb[:, :, 0] + weight_g * img_rgb[:, :, 1] + weight_b * img_rgb[:, :, 2]
            jbf_out = JBF.joint_bilateral_filter(img_rgb, img_y.astype(np.uint8)).astype(np.uint8)
            cv2.imwrite('./output_image/img_y_{}_{}.png'.format(test_img, i), img_y)
        else:
            jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray.astype(np.uint8)).astype(np.uint8)
            cv2.imwrite('./output_image/img_gray_{}.png'.format(test_img), img_gray)

        jbf_out_rgb, jbf_out = np.int32(jbf_out_rgb), np.int32(jbf_out)
        diff = jbf_out_rgb - jbf_out 
        cost = np.sum(abs(diff))
        cv2.imwrite('./output_image/img_out_{}_{}_{}.png'.format(test_img, i, cost), cv2.cvtColor(jbf_out.astype(np.float32), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()