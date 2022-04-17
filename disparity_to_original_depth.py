import numpy as np
import cv2
import argparse
import sys
import tifffile as tiff


import numpy as np
from scipy import interpolate

def interpolate2d(array):
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    #mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    out = interpolate.griddata((x1, y1), newarr.ravel(),(xx, yy),method='cubic')
    return out




def disp_to_original_depth(disparity_path, R1, Q, K1, D1):
    
    # ----------------------------------------------------------------------
    # load the disparity image
    disp = cv2.imread(str(disparity_path),-1)
    if disp.dtype == np.uint16: 
        # assuming that disparities are stored as uit16 integers multiplied by 128 ( the disparity inference script should save them in this way)
        disp = disp.astype(np.float32)/128.
    else:
        disp = disp.astype(np.float32)
    
    
    h,w = disp.shape[:2]
    # ----------------------------------------------------------
    # reproject disparity to 3D in the left rectified frame of reference 
    img_3d = cv2.reprojectImageTo3D(	disp, Q)
    
    # ------------------------------------------------------------------
    # rotate the reconstructed point cloud to the original left frame of reference
    RT = np.eye(4)
    RT[:3,:3]= np.linalg.inv(R1)
    
    img_3d[img_3d[:, :, 2] == 0] = np.nan
    
    # vectorize and keep only known points
    ptcloud = img_3d.reshape(-1, 3)
    ptcloud = ptcloud[~np.isnan(ptcloud).any(axis=1)]
    
    # convert ptcloud to homogeneous coordinates
    ptcloud_h = np.hstack((ptcloud, np.ones((ptcloud.shape[0], 1))))
    # rotate it to the left rectified view
    
    ptcloud_h = (RT@ ptcloud_h.T).T
    
    # convert back from homogeneous
    ptcloud = ptcloud_h[:,:3]/(ptcloud_h[:,3].reshape(-1,1))

    # -------------------------------------------------------------------------
    # project images back to original frame of reference 
    
    # possibly you can do the above rotation within the projecPoints but I haven't tested it.
    pix_loc, _ = cv2.projectPoints(	ptcloud, np.zeros(3), np.zeros(3), K1, D1)

    pix_loc = pix_loc.reshape(-1,2)
    
    
    valid_indexes = (
        (pix_loc[:, 0] >= 0)
        & (pix_loc[:, 0] < w)
        & (pix_loc[:, 1] >= 0)
        & (pix_loc[:, 1] < h)
    )
    depth_idxs = pix_loc[valid_indexes].astype(int)
    valid_depths = ptcloud[valid_indexes]
    xs, ys = depth_idxs[:, 0], depth_idxs[:, 1]
    
    # create the disparity map (float), zero values for unknown disparities
    target_img_3d = np.zeros((h,w,3))
    target_img_3d[ys, xs] = valid_depths
    
    
    
    return target_img_3d


def main(args):
    cv2.namedWindow('disparity', 2)
    cv2.namedWindow('disp_to_original_depth', 2)
    cv2.namedWindow('interpolated_depth', 2)
    # rectification rotation (R1) and projection matrices (P1,P2) obtained from cv2.stereoRectify()
    K1 = np.array([  1.03530811e+03, 0.0, 5.96955017e+02, 0.0,
            1.03508765e+03, 5.20410034e+02, 0.0, 0.0, 1.0  ]).reshape(3,3)
    D1 = np.array([ -5.95157442e-04, -5.46629308e-04, 0.0, 0.0,
            1.82959007e-03 ]).reshape(1,5)    

    R1 = np.array([ 9.9998331608991398e-01, 5.7682930542439156e-03,
            3.0714338576628622e-04, -5.7682732817294444e-03,
            9.9998336124626230e-01, -6.5222500481189457e-05,
            -3.0751449777963278e-04, 6.3449725329078383e-05,
            9.9999995070448167e-01 ]).reshape(3,3)
    
    Q = np.array([1.0, 0.0, 0.0, -6.4145299911499023e+02, 0.0, 1.0, 0.0,
            -5.2002098083496094e+02, 0.0, 0.0, 0.0,
            1.0350333251953125e+03, 0.0, 0.0, 2.4134424391349485e-01,
            0.0]).reshape(4,4)
    
    
    print('THE SCRIPT SERVES AS A SAMPLE AND USES HARDCODED CALIBRATION PARAMETERS FROM DATASET ONE KEYFRAME 1 DATASET 1')
    print('PLEASE ADAPT IT TO YOUR OWN PIPELINE')

    try:
        img3d = disp_to_original_depth(args.disp, R1, Q, K1, D1)
    except FileNotFoundError as e:
        print(e)
        return 1
    
    # -------------------------------------------------------------
    # interpolate the missing values
    # the interpolate2d would interpolate only the nan values
    interpolated_depth = img3d[...,-1].copy()
    interpolated_depth[interpolated_depth==0]=np.nan
    interpolated_depth = interpolate2d(interpolated_depth)
    
    
    if args.original_depth:
        # compare the predicted and original
        gt_depth = tiff.imread(args.original_depth)[...,-1]
        assert gt_depth.shape == interpolated_depth.shape
        gt_depth[gt_depth==0]=np.nan
        diff = (gt_depth-interpolated_depth)
        diff = np.abs(diff)
        depth_error = np.nanmean(diff)
        print(f'MAE: {depth_error}')# Mean Absolute Error

    cv2.imshow('disp_to_original_depth', img3d[...,-1].astype(np.uint8))
    cv2.imshow('interpolated_depth', interpolated_depth.astype(np.uint8))
    cv2.imshow('disparity', cv2.imread(str(args.disp)))
    cv2.waitKey(0)
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('disp', help = 'path to disparity file ')
    parser.add_argument('--original_depth', help='path to original depth for comparison')
    sys.exit(main(parser.parse_args()))
