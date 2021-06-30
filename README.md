===========================================
MICCAI 2019 SCARED challenge, DeepPruner submission.
===========================================

This repository contains code used to participate in the MICCAI 2019 SCARED challenge using DeepPruner.

The code base is a fork of [DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch](https://arxiv.org/abs/1909.05845)
and contains additional code to recreate results submitted in the MICCAI 2019 SCARED challenge.

The majority of the work was done on the data, hence the DeepPruner codebase is almost untouched.

.. contents::

Citations
---------

If you find code in this repository useful, please consider citing:

The original authors of DeepPruner:

```bibtex
@inproceedings{duggal2019deeppruner,
  title={Deeppruner: Learning efficient stereo matching via differentiable patchmatch},
  author={Duggal, Shivam and Wang, Shenlong and Ma, Wei-Chiu and Hu, Rui and Urtasun, Raquel},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4384--4393},
  year={2019}
}
```

The MICCAI 2019 SCARED challenge:

```bibtex
@article{allan2021stereo,
  title={Stereo Correspondence and Reconstruction of Endoscopic Data Challenge},
  author={Allan, Max and Mcleod, Jonathan and Wang, Cong Cong and Rosenthal, Jean Claude and Fu, Ke Xue and Zeffiro, Trevor and Xia, Wenyao and Zhanshi, Zhu and Luo, Huoling and Zhang, Xiran and others},
  journal={arXiv preprint arXiv:2101.01133},
  year={2021}
}
```

Dataset
-------

The SCARED challenge made available an endoscopic dataset with semi-dense depth annotations
and made possible training learning based methods for depth and disparity estimation
in the surgical domain. The dataset consists of stereo video sequences, accompanied with
ground truth point map annotations and stereo calibration parameters. In order
for the dataset to be used with stereo matching methods, like DeepPruner, we
first need to generate appropriate stereo rectified input frames and disparity
samples. The former can be easily achieved using the provided calibration
parameters and OpenCV stereo rectification codebase, while the later can be achieved by
projecting the ground truth point cloud into the rectified views and measuring the
displacement in x (disparity) between the projection of points in the left and right views.

With disparity version of SCARED generated, one can use this repository to recreate
the challenge's submission results(DeepPrunner)

Data manipulation code will be posted soon, for now you can follow the steps
described bellow and create a keyframe only dataset.

Data pre-processing
-------------------

In order for the SCARED data to be used with stereo matching approaches we need
to generate disparity samples from the the provided point maps. We process the data
the following the step bellow:

1. Extend the calibration parameters provided for each sub dataset by computing
the rotation matrices(R1,R2), projection matrices(P1,P2) for the stereo
rectifed views and the Q matrix. This can be achieved using OpenCV's
`stereoRectify()` function.

2. Stereo rectify RGB frames using OpenCV's `initUndistortRectifyMap`, `remap`
functions and the matrices computed during step one.

3. Using P1, P2, project the corresponding pointcloud to both left and right stereo rectified
image frames. Disparity is the difference in x between the location of the two
projections and is stored in the projection location of the
left image. Doing this for all points in the pointcloud, gives us a disparity
image. Because projections will not end up in discrete pixel coordinates, we
first compute the horizontal distance between the two projection in pixels
and then store the disparity in the rounded projection coordinated in the
left image. This will introduce additional error to the ground truth disparity
samples of 0.5 pixels at most.

4. Store the disparity images as uint16 pngs following the KITTI format. In our
case, the scale factor will be 128 (instead of 256)in order to preserve large
disparity information for objects close to the camera.

5. Save the RGB and disparity samples according to the following tree structure:

```tree
SCARED_disparity_keyframe_dataset
├── clean_disparity_128
│   ├── 1_1.png
│   ├── 1_2.png
│   ├── 1_3.png
|   ├── .......
│   ├── 7_4.png
│   └── 7_5.png
├── rect_left
│   ├── 1_1.png
│   ├── 1_2.png
│   ├── 1_3.png
|   ├── .......
│   ├── 7_4.png
│   └── 7_5.png
└── rect_right
    ├── 1_1.png
    ├── 1_2.png
    ├── 1_3.png
    ├── .......
    ├── 7_4.png
    └── 7_5.png
```

The naming convention followed, is {dataset number}_{keyframe_number}.png
To recreate the SCARED fine-tuned model you should stereo-rectify the provided
frames using alpha=0 as a parameter to the `stereoRectify()` functions.
The same RGB stereo rectification process must be followed for the test set but
with alpha=1 in order to preserve the whole rgb frame within the rectified view.

Pre-trained models
-----------------

The SCARED finetuned model, used for the challenge submission can be found [here](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabdps_ucl_ac_uk/EcRDTpJcmFxGsU9nNYcTFEQBbnEWnS0h2OUSlol7ynnzYQ?e=kd5K6p)

To recreate it, you need to fine-tune DeepPruner on SCARED data, starting from the sceneflow model
available at the [Original DeepPruner repository](https://github.com/uber-research/DeepPruner)

SCARED Fine-tuning
-----------------

1. Prepare the stereo rectified keyframe dataset as described above.
2. Download the sceneflow pre-trained model from the [Original DeepPruner repository](https://github.com/uber-research/DeepPruner)
3. Modify the /deeppruner/scared_training_config.json to include paths to both
the disparity dataset root folder and the sceneflow pre-trained model.
4. navigate to the /deeppruner subdirectory of this repository and run the training script

```code
python finetune_scared.py --config ./scared_training_config.json
```

SCARED submission
-----------------

1. Stereo rectify the videos(alpha=1) and store them as images, following the file structure
bellow(dataset 8, 9 and keyframes 1-5). In case of keyframe_5, where
there isn't any video sequence, you only use the keyframe information instead.

```tree
SCARED_test_dataset
├── left_rect
│   └── dataset_x
│       └── keyframe_y
│           ├── frame000000.png
│           ├── frame000001.png
|           ├── ...............
│           └── frame999999.png
└── right_rect
    └── dataset_x
        └── keyframe_y
            ├── frame000000.png
            ├── frame000001.png
            ├── ...............
            └── frame999999.png

```

2. Navigate to ./deeppruner/ subdirectory and run the inference script

```code

python submission_scared.py --datapath /path/to/SCARED_test_dataset --loadmodel /path/to/scared/finetuned/model --savedir /path/to/store/inferred/disparities/

```

The network will predict disparity maps for every input frame stored under the
SCARED_test_dataset folder.

Data post-processing
--------------------

In oder to evaluate output against the SCARED ground truth, we need to convert
inferred disparities back to the scared format and more importantly in the
original left frame of reference including camera distortions. To do that we
have to reconstruct inferred disparities using the Q matrices, obtained during the
calibration process, and OpenCV's `reprojectImageTo3D()`. This will give us a
pointmap expressed in the left rectified frame of reference. We then need to convert the
pointmap to pointclouds, rotate them by inverse R1( obtained along Q during stereo
rectification) and finally create the final pointmaps, by project the rotated pointcloud
back to the original frame of reference. This is done by storing xyz location at the rounded
projection coordinates. Because of the rectification alpha used for the test frames,
this process results in pointmaps with a grid of unknown values. The last step is to interpolate
the missing values using cubic interpolation.
