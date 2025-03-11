
import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        
        h, w = img.shape[:2]
        normalize = np.zeros((h, w))
        output = np.zeros((h, w, 3))
        expand_size = self.wndw_size ** 2

        ### TODO ###
        # Spatial Kernel G_s
        G_s = np.exp(-0.5 * np.sum((np.indices((self.wndw_size, self.wndw_size)) - (self.wndw_size // 2)) ** 2, axis=0) / (self.sigma_s ** 2))
        
        # Range Kernel Weight
        r_weight =  np.exp(-0.5 * (np.linspace(0, 1, 256) / self.sigma_r)**2)
        
        # Range Kernel G_r
        for i in range(self.wndw_size):
            for j in range(self.wndw_size):
                
                intensity_diff = np.abs(guidance - padded_guidance[i:i+h, j:j+w])
                if guidance.ndim == 3:
                    G_r = r_weight[intensity_diff[:, :, 0]] * r_weight[intensity_diff[:, :, 1]] * r_weight[intensity_diff[:, :, 2]]
                else:
                    G_r = r_weight[intensity_diff]

                joint_weight = G_r * G_s[i, j]
                normalize += joint_weight
                
                # output += np.expand_dims(joint_weight, axis=2) * padded_img[i:i+h, j:j+w]
                for k in range(3):
                    output[:, :, k] += joint_weight * padded_img[i:i+h, j:j+w, k]
        
        output /= np.expand_dims(normalize, axis=2)
        
        return np.clip(output, 0, 255).astype(np.uint8)