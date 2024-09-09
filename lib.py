import cv2
import numpy as np
import skimage as sk
import skimage.io as skio
from skimage import img_as_ubyte
import itertools
import os
from tqdm import tqdm
from skimage.transform import rescale

def metric(im1, im2):
    # Flatten the images to 1D vectors
    im1_flat = im1.flatten()
    im2_flat = im2.flatten()

    # Normalize the vectors (divide by their L2 norms)
    im1_norm = im1_flat / np.linalg.norm(im1_flat)
    im2_norm = im2_flat / np.linalg.norm(im2_flat)

    # Compute the dot product (NCC)
    ncc = np.dot(im1_norm, im2_norm)
    return ncc

def sobel_filter(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in x direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in y direction
    sobel_image = np.hypot(sobel_x, sobel_y)  # Combine the two directions
    return sobel_image

def crop_image(im, X):
    # Get image dimensions
    height, width = im.shape

    # Calculate number of pixels to crop based on the percentage X
    crop_pixels_h = int(height * X / 2)  # crop equally from top and bottom
    crop_pixels_w = int(width * X / 2)   # crop equally from left and right

    # Perform cropping
    cropped_im = im[crop_pixels_h:height-crop_pixels_h, crop_pixels_w:width-crop_pixels_w]

    return cropped_im

def create_gaussian_pyramid(im1, im2, min_resolution = (512, 512)):
    """
    Perform a Gaussian pyramid downsampling on two images until either the width or height
    is less than the specified minimum shape, and return the pyramid in reverse order.

    Parameters:
        im1 (ndarray): First input image (single channel).
        im2 (ndarray): Second input image (single channel).
        min_shape (tuple): Minimum (height, width) for the image resolution before stopping downscaling.

    Returns:
        pyramid_levels (list of tuples): List of tuples containing downsampled versions 
                                         of im1 and im2 at each level in reverse order 
                                         (smallest image first).
    """
    # Check if the image is already smaller than or equal to the minimum shape
    pyramid_levels = []  # List to store downsampled image pairs
    current_im1, current_im2 = im1, im2

    # Step 1: Continue downsampling until width or height is below min_shape
    while current_im1.shape[0] > min_resolution[0] and current_im1.shape[1] > min_resolution[1]:
        # Add the current level to the pyramid
        pyramid_levels.append((current_im1, current_im2))
        
        # Downsample both images by a factor of 0.5
        current_im1 = rescale(current_im1, 0.5, anti_aliasing=True)
        current_im2 = rescale(current_im2, 0.5, anti_aliasing=True)

    # Add the last downsampled image which is smaller than min_shape
    pyramid_levels.append((current_im1, current_im2))

    # Return pyramid levels in reverse order (smallest first)
    return pyramid_levels[::-1]


def align(im1, im2, D=30, min_resolution = (128, 128)):

    best_offset = (0, 0)
    best_score = -np.inf  # Initialize to a low value for maximizing NCC

    total_pixels = im1.shape[0] * im1.shape[1]

    print(f"Total number of pixels in im1 (excluding channels): {total_pixels}")

    gaussian_pyramid = create_gaussian_pyramid(im1, im2, min_resolution=min_resolution) 
    if len(gaussian_pyramid) == 0:
        gaussian_pyramid = [(im1, im2)]

    # Differences
    diff_x = range(-D // 2, D // 2 + 1)
    diff_y = range(-D // 2, D // 2 + 1)
    
    print("Pyramid size", len(gaussian_pyramid))

    for i, (img1, img2) in enumerate(gaussian_pyramid): 

        prod = list(itertools.product(diff_x, diff_y))
        best_offset = (2 * best_offset[0], 2 * best_offset[1])

        print(f"Iteration {i}, Shape of images: {img1.shape} offset: {best_offset} Search X: {diff_x} Search Y: {diff_y}")

        # Preprocess with the Sobel filter
        sampled_im1, sampled_im2 = sobel_filter(img1), sobel_filter(img2)

        # Shift image by initial offset already produced
        shifted_im1 = np.roll(np.roll(sampled_im1, best_offset[0], axis=1), best_offset[1], axis=0)

        for (a, b) in tqdm(prod, total=len(prod)):

            # Shift sampled_im1 by 'a' pixels horizontally and 'b' pixels vertically
            shifted_current_im1 = np.roll(np.roll(shifted_im1, a, axis=1), b, axis=0)

            # Compute NCC for this trial
            score = metric(shifted_current_im1, sampled_im2)

            if score > best_score:
                best_score = score
                best_offset = (a, b)

        # Update the differences range in X and Y
        diff_x = range(best_offset[0] - 1, best_offset[0] + 1)
        diff_y = range(best_offset[1] - 1, best_offset[1] + 1)


    # After finding the best offset, apply the shift
    aligned_im1 = np.roll(np.roll(im1, best_offset[0], axis=1), best_offset[1], axis=0)

    print('Returning', best_offset, best_score)
    return aligned_im1, best_offset, best_score


def run(imname, difference_range = 30, crop_percent = 0.15, min_resolution = (128, 128)):

    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)
    im = sk.img_as_float(im)

    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.uint32)

    print(height)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    height = abs(int(height))

    r = crop_image(r, crop_percent) 
    g = crop_image(g, crop_percent) 
    b = crop_image(b, crop_percent) 

    difference_range = 30
    ag, green_offset, score_g = align(g, b, D = difference_range, min_resolution=min_resolution)
    ar, red_offset, score_r = align(r, b, D = difference_range, min_resolution=min_resolution)

    print('Red Shape', ar.shape)
    print('Best offset', red_offset)

    print('Green Shape', ag.shape)
    print('Best offset', green_offset)
    print('Blue Shape', b.shape)

    # create a color image
    im_out = np.dstack([ar, ag, b])

    # convert to uint8 for saving/displaying
    im_out_uint8 = img_as_ubyte(im_out)

    # save the image
    fname = f'out_{imname}'
    skio.imsave(fname, im_out_uint8)

    # display the image
    skio.imshow(im_out_uint8)
    skio.show()
