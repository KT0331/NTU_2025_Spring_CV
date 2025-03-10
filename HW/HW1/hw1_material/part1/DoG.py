import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        ref_image = image
        for i in range(0, self.num_octaves):
            gaussian_images.append(ref_image)
            for j in range(1, self.num_guassian_images_per_octave):
                filter_sigma = self.sigma**j
                Gauss_Blur_image = cv2.GaussianBlur(ref_image, ksize = (0, 0), sigmaX = filter_sigma, sigmaY = filter_sigma)
                gaussian_images.append(Gauss_Blur_image)
            # dsize in cv2.resize means scale assignd by fx & fy
            ref_image = cv2.resize(gaussian_images[-1], (0, 0), fx = 1/2, fy = 1/2, interpolation = cv2.INTER_NEAREST)
                
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(0, self.num_octaves):
            index = i * self.num_guassian_images_per_octave
            for j in range(0, self.num_DoG_images_per_octave):
                substacted_image = np.subtract(gaussian_images[index+j+1], gaussian_images[index+j])
                dog_images.append(substacted_image)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        index = 0
        y_size = len(image)
        x_size = len(image[0])
        keypoints = []

        for octave in range(0, self.num_octaves):
            for num_dog in range(1 + index, self.num_DoG_images_per_octave + index - 1):
                for y in range(1, y_size-1):
                    for x in range(1, x_size-1):
                        is_local_extremum = False
                        upper = dog_images[num_dog+1][y-1:y+2, x-1:x+2]
                        middle = dog_images[num_dog][y-1:y+2, x-1:x+2]
                        lower = dog_images[num_dog-1][y-1:y+2, x-1:x+2]
                        entry_list = np.concatenate((upper, middle, lower), axis = 0)
                        max_entry = np.amax(entry_list)
                        min_entry = np.amin(entry_list)
                        center = middle[1][1]
                        if np.abs(center) > np.abs(self.threshold):
                            is_local_extremum = (center == max_entry) or (center == min_entry)      

                        if is_local_extremum:
                            pixel = (y * (2**octave), x * (2**octave))
                            keypoints.append(pixel)
                           

            index = index + self.num_DoG_images_per_octave
            y_size = y_size // 2
            x_size = x_size // 2

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis = 0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]
        
        return keypoints, dog_images
        # return keypoints

if __name__ == '__main__':
    img = cv2.imread('./testdata/1.png', 0).astype(np.float64)    
    DoG = Difference_of_Gaussian(threshold = 5.0)
    
    # find keypoint from DoG and sort it
    keypoints = DoG.get_keypoints(img)
