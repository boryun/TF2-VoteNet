# OverAll

This is a TensorFlow2 implementation of [VoteNet](https://openaccess.thecvf.com/content_ICCV_2019/html/Qi_Deep_Hough_Voting_for_3D_Object_Detection_in_Point_Clouds_ICCV_2019_paper.html), the official PyTorch implementation is [facebookresearch/votenet](https://github.com/facebookresearch/votenet), I made some simplification and make it available for TF2.

With almost the same training configuration, I got consistent results on ScanNet and SUNRGBD datasets (mean AP 60.6@0.25 and 33.6@0.5 on ScanNet, 57.9@0.25 and 31.7@0.5 on SUNRGBD, see [Eval Result](#eval-result) for more detail).

# Train & Eval

## Environment

The following env config is tested with python-3.7.9 and CUDA-10.1:

``` text
tensorflow==2.3.1
plyfile
scipy
pandas
opencv-python
tqdm
open3d (optional, only for visualization)
```

## Data Preparation

**ScanNet**

1. Download [ScanNet](http://www.scan-net.org/)/[ScanNet GitHub](https://github.com/ScanNet/ScanNet) datasets;

2. Run `scannet_utils.py` under `utils/dataset_utils/ScanNet` (change `SCANNET_DIR` under `__main__` part to the absolute path to scans folder);

3. If everything is ok, you can use `show_scannet_gt` functions inside `scannet_utils.py` to visualize.

**SUNRGBD**

1. Download and unzip [SUNRGBD](http://rgbd.cs.princeton.edu/data/) datasets (SUNRGBD.zip, SUNRGBDtoolbox.zip, SUNRGBDMeta2DBB_v2.mat, SUNRGBDMeta3DBB_v2.mat);

2. Run `extract_data.m` under `utils/dataset_utils/SUNRGBD` (you may need to change `dataset_root` and `output_root` variables first);

3. Run `sunrgbd_utils.py` under `utils/dataset_utils/SUNRGBD` (change `EXTRACT_SUNRGBD_DIR` under `__main__` part to the same as `output_root` in `extract_data.m`);

4. If everything is ok, you can use `show_extracted_scene` and `show_sunrgbd_gt` functions inside `sunrgbd_utils.py` to visualize.

## Build Ops

The CUDA Ops is modified from [charlesq34/pointnet2](https://github.com/charlesq34/pointnet2). To build required Ops, first activate conda environment with TensorFlow2 installed and make sure `CUDA_HOME` variable is set in bash env. Then, run `compile_op.sh` under `model/custom_op`. If everything is ok, you could test each OP by run `op_interface.py` under each subfolders.

## Train & Eval

To train on ScanNet, run `python -B train.py --gpus 0 --dataset scannet --batch_size 16 --log_dir logs/scannet --verbose_interval 15 --keep_all_classes`, to train on SUNRGBD, run `python -B train.py --gpus 0 --dataset sunrgbd --batch_size 16 --log_dir logs/sunrgbd --verbose_interval 45 --keep_all_classes`.

To evaluate on ScanNet, run `python -B test.py --gpus 0 --dataset scannet --batch_size 32 --log_dir logs/scannet --keep_all_classes`, change `--dataset` and `--log_dir` args to evaluate on SUNRGBD.

# Eval Result

The official implementation set `per_class_proposal` flags default to True during training and testing (which is denoted as `keep_all_classes` in this project). That is, for each prediction with objectness greater than a specific threshold, instead of assigning it to the semantic class with maximum predict probs, we make it a prediction to all classes with class score equal to `objectness * class_prob` (i.e. split each prediction into num_class proposals). Which is a little tricky but could improve meanAP about 1~2 points.

The result under each circumstance is listed following.

## ScanNet

**`keep_all_classes=False`**
``` text
                   >>> AP @ 0.25 <<<                    
basic - mean_P: 0.2321, mean_R: 0.7465, mean_AP: 0.5833
detail - :
     classname  	precision	 recall  	AP @0.25
--------------------------------------------------------------------
       cabinet: 	0.132877 	0.731183 	0.417338
           bed: 	0.622807 	0.876543 	0.859928
         chair: 	0.347199 	0.915205 	0.878332
          sofa: 	0.407583 	0.886598 	0.781550
         table: 	0.229358 	0.785714 	0.642076
          door: 	0.153774 	0.706638 	0.477652
        window: 	0.134673 	0.641844 	0.394230
     bookshelf: 	0.156915 	0.766234 	0.530085
       picture: 	0.026353 	0.252252 	0.075146
       counter: 	0.119883 	0.788462 	0.393567
          desk: 	0.203237 	0.889764 	0.674087
       curtain: 	0.217647 	0.552239 	0.456462
 refridgerator: 	0.197080 	0.473684 	0.409042
shower curtain: 	0.181159 	0.892857 	0.733259
        toilet: 	0.354430 	0.965517 	0.945928
          sink: 	0.133829 	0.734694 	0.592867
       bathtub: 	0.432836 	0.935484 	0.902609
otherfurniture: 	0.126535 	0.641509 	0.335410

                    >>> AP @ 0.5 <<<                    
basic - mean_P: 0.1662, mean_R: 0.4736, mean_AP: 0.3365
detail - :
     classname  	precision	 recall  	AP @0.50
--------------------------------------------------------------------
       cabinet: 	0.055203 	0.303763 	0.090350
           bed: 	0.578947 	0.814815 	0.764560
         chair: 	0.280366 	0.739035 	0.637291
          sofa: 	0.369668 	0.804124 	0.658194
         table: 	0.166806 	0.571429 	0.414194
          door: 	0.082479 	0.379015 	0.187305
        window: 	0.047619 	0.226950 	0.106432
     bookshelf: 	0.111702 	0.545455 	0.363880
       picture: 	0.003294 	0.031532 	0.002732
       counter: 	0.049708 	0.326923 	0.065259
          desk: 	0.134892 	0.590551 	0.266937
       curtain: 	0.100000 	0.253731 	0.191224
 refridgerator: 	0.145985 	0.350877 	0.295723
shower curtain: 	0.050725 	0.250000 	0.103138
        toilet: 	0.329114 	0.896552 	0.864321
          sink: 	0.063197 	0.346939 	0.181405
       bathtub: 	0.358209 	0.774194 	0.752500
otherfurniture: 	0.062895 	0.318868 	0.111581
```

**`keep_all_classes=True`**
``` text
                   >>> AP @ 0.25 <<<                    
basic - mean_P: 0.0111, mean_R: 0.8357, mean_AP: 0.6062
detail - :
     classname  	precision	 recall  	AP @0.25
--------------------------------------------------------------------
       cabinet: 	0.016328 	0.782258 	0.418686
           bed: 	0.004264 	0.938272 	0.894630
         chair: 	0.071429 	0.930556 	0.879351
          sofa: 	0.005330 	0.979381 	0.895526
         table: 	0.016833 	0.857143 	0.642298
          door: 	0.019470 	0.743041 	0.491086
        window: 	0.011222 	0.709220 	0.421896
     bookshelf: 	0.003872 	0.896104 	0.617165
       picture: 	0.003142 	0.252252 	0.081045
       counter: 	0.002637 	0.903846 	0.420516
          desk: 	0.006789 	0.952756 	0.635765
       curtain: 	0.002749 	0.731343 	0.428705
 refridgerator: 	0.003086 	0.964912 	0.480167
shower curtain: 	0.001459 	0.928571 	0.634183
        toilet: 	0.003198 	0.982759 	0.956755
          sink: 	0.004321 	0.785714 	0.646382
       bathtub: 	0.001627 	0.935484 	0.921928
otherfurniture: 	0.022893 	0.769811 	0.445044

                    >>> AP @ 0.5 <<<                    
basic - mean_P: 0.0073, mean_R: 0.5181, mean_AP: 0.3459
detail - :
     classname  	precision	 recall  	AP @0.50
--------------------------------------------------------------------
       cabinet: 	0.007070 	0.338710 	0.068154
           bed: 	0.004096 	0.901235 	0.843692
         chair: 	0.057794 	0.752924 	0.658550
          sofa: 	0.004657 	0.855670 	0.731059
         table: 	0.012513 	0.637143 	0.426769
          door: 	0.010549 	0.402570 	0.183634
        window: 	0.004264 	0.269504 	0.135346
     bookshelf: 	0.002749 	0.636364 	0.405833
       picture: 	0.000505 	0.040541 	0.005502
       counter: 	0.000954 	0.326923 	0.087288
          desk: 	0.004769 	0.669291 	0.326690
       curtain: 	0.001178 	0.313433 	0.174414
 refridgerator: 	0.002020 	0.631579 	0.302148
shower curtain: 	0.000505 	0.321429 	0.122931
        toilet: 	0.002637 	0.810345 	0.740417
          sink: 	0.001852 	0.336735 	0.213620
       bathtub: 	0.001178 	0.677419 	0.637472
otherfurniture: 	0.012008 	0.403774 	0.162357
```

## SUNRGBD

**`keep_all_classes=False`**
``` text
                   >>> AP @ 0.25 <<<                    
basic - mean_P: 0.1424, mean_R: 0.7625, mean_AP: 0.5617
detail - :
     classname  	precision	 recall  	AP @0.25
--------------------------------------------------------------------
           bed: 	0.244330 	0.872928 	0.823099
         table: 	0.141400 	0.754192 	0.481973
          sofa: 	0.151773 	0.810642 	0.639127
         chair: 	0.205981 	0.867904 	0.749853
        toilet: 	0.135545 	0.947020 	0.893921
          desk: 	0.164282 	0.423512 	0.209793
       dresser: 	0.062636 	0.666667 	0.248601
   night_stand: 	0.120475 	0.838583 	0.616847
     bookshelf: 	0.097436 	0.635451 	0.292709
       bathtub: 	0.100000 	0.807692 	0.661458

                    >>> AP @ 0.5 <<<                    
basic - mean_P: 0.0846, mean_R: 0.4534, mean_AP: 0.3068
detail - :
     classname  	precision	 recall  	AP @0.50
--------------------------------------------------------------------
           bed: 	0.169588 	0.605893 	0.512608
         table: 	0.066789 	0.356237 	0.157172
          sofa: 	0.111925 	0.597809 	0.454691
         chair: 	0.148128 	0.624141 	0.489126
        toilet: 	0.103318 	0.721854 	0.641376
          desk: 	0.061057 	0.157403 	0.042096
       dresser: 	0.036538 	0.388889 	0.107836
   night_stand: 	0.072964 	0.507874 	0.351540
     bookshelf: 	0.023077 	0.150502 	0.034335
       bathtub: 	0.052381 	0.423077 	0.276779
```

**`keep_all_classes=True`**
``` text
                   >>> AP @ 0.25 <<<                    
basic - mean_P: 0.0194, mean_R: 0.8577, mean_AP: 0.5794
detail - :
     classname  	precision	 recall  	AP @0.25
--------------------------------------------------------------------
           bed: 	0.006948 	0.950276 	0.838020
         table: 	0.028090 	0.853170 	0.486100
          sofa: 	0.007797 	0.906103 	0.656206
         chair: 	0.119080 	0.868493 	0.734528
        toilet: 	0.001980 	0.973510 	0.901051
          desk: 	0.021855 	0.798328 	0.247470
       dresser: 	0.002330 	0.800926 	0.310448
   night_stand: 	0.003097 	0.905512 	0.639489
     bookshelf: 	0.002639 	0.655518 	0.288995
       bathtub: 	0.000606 	0.865385 	0.691415

                    >>> AP @ 0.5 <<<                    
basic - mean_P: 0.0122, mean_R: 0.4945, mean_AP: 0.3176
detail - :
     classname  	precision	 recall  	AP @0.50
--------------------------------------------------------------------
           bed: 	0.004821 	0.659300 	0.514884
         table: 	0.013385 	0.406544 	0.193741
          sofa: 	0.005427 	0.630673 	0.477904
         chair: 	0.085657 	0.624730 	0.506742
        toilet: 	0.001495 	0.735099 	0.599025
          desk: 	0.007312 	0.267093 	0.050333
       dresser: 	0.001239 	0.425926 	0.144883
   night_stand: 	0.001980 	0.578740 	0.378800
     bookshelf: 	0.000781 	0.193980 	0.050044
       bathtub: 	0.000296 	0.423077 	0.259780
```
