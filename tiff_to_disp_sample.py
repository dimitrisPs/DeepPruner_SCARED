import numpy as np
import cv2
import tifffile as tiff
import argparse
import sys


def tiff_to_disp(tiff_path, R1, P1, P2):
    
    
    img_3d = tiff.imread(str(tiff_path))
    h,w = img_3d.shape[:2]
    if h>1024: # if we read samples from the tar.gz sequence 
        # point images for the left and right views are stacked vertically
        # we only need points from the left view
        img_3d = img_3d[:h//2] 
    # items with z=0 are unknown, set them to nan.
    img_3d[img_3d[:, :, 2] == 0] = np.nan
    
    
    # vectorize and keep only known points
    ptcloud = img_3d.reshape(-1, 3)
    ptcloud = ptcloud[~np.isnan(ptcloud).any(axis=1)]
    
    # convert ptcloud to homogeneous coordinates
    ptcloud_h = np.hstack((ptcloud, np.ones((ptcloud.shape[0], 1))))
    # rotate it to the left rectified view
    RT = np.eye(4)
    RT[:3,:3]= R1
    ptcloud_h = (RT@ ptcloud_h.T).T
    
    #project points to both views and convert back from homogeneous coordinates
    projected_left_h = (P1 @ ptcloud_h.T).T
    projected_right_h = (P2 @ ptcloud_h.T).T
    projected_left = projected_left_h[:, :2] / projected_left_h[:, 2].reshape(-1, 1)
    projected_right = projected_right_h[:, :2] / projected_right_h[:, 2].reshape(-1, 1)
    projected_left = projected_left.reshape(-1,2)
    projected_right = projected_right.reshape(-1,2)

    # compute the disparity values for each known point
    disparities = (projected_left - projected_right)[:, 0]
    
    #find the discrete locations of the projections in the left image that end up inside the image
    
    valid_indexes = (
        (projected_left[:, 0] >= 0)
        & (projected_left[:, 0] < w)
        & (projected_left[:, 1] >= 0)
        & (projected_left[:, 1] < h)
    )
    disparity_idxs = projected_left[valid_indexes].astype(int)
    valid_disparities = disparities[valid_indexes]
    xs, ys = disparity_idxs[:, 0], disparity_idxs[:, 1]
    
    # create the disparity map (float), zero values for unknown disparities
    disparity_map = np.zeros((h,w))
    disparity_map[ys, xs] = valid_disparities
    return disparity_map


def main(args):
    cv2.namedWindow('disparity', 2)
    
    # rectification rotation (R1) and projection matrices (P1,P2) obtained from cv2.stereoRectify()
    R1 = np.array([ 9.9998331608991398e-01, 5.7682930542439156e-03,
            3.0714338576628622e-04, -5.7682732817294444e-03,
            9.9998336124626230e-01, -6.5222500481189457e-05,
            -3.0751449777963278e-04, 6.3449725329078383e-05,
            9.9999995070448167e-01 ]).reshape(3,3)
    P1 = np.array([ 1.0350333251953125e+03, 0.0, 6.4144980239868164e+02,
            0.0, 0.0, 1.0350333251953125e+03, 5.2002452087402344e+02,
            0.0, 0.0, 0.0, 1.0, 0.0 ]).reshape(3,4)
    
    P2 = np.array([ 1.0350333251953125e+03, 0.0, 6.4144980239868164e+02,
            -4.2886182343188593e+03, 0.0, 1.0350333251953125e+03,
            5.2002452087402344e+02, 0.0, 0.0, 0.0, 1.0, 0.0 ]).reshape(3,4)
    
    print('THE SCRIPT SERVES AS A SAMPLE AND USES HARDCODED CALIBRATION PARAMETERS FROM DATASET ONE KEYFRAME 1 DATASET 1')
    print('PLEASE ADAPT IT TO YOUR OWN PIPELINE')

    try:
        disparity_map = tiff_to_disp(args.tiff, R1, P1, P2)
    except FileNotFoundError as e:
        print(e)
        return 1

    cv2.imshow('disparity', disparity_map.astype(np.uint8))
    cv2.waitKey(0)
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tiff', help = 'path to tiff file ')
    sys.exit(main(parser.parse_args()))
