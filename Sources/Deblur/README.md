[Paper Link](https://arxiv.org/pdf/1907.13106.pdf)

## Prerequisites:
1. Linux
2. Python 2 or 3
3. Pytorch version 0.4.1
4. CPU or NVIDIA GPU + CUDA CuDNN (CUDA 8.0)

    - pip install Pillow==6.2.2
    - pip install torch==0.4.1
    - pip install torchvision== 0.2.1


## To test Deblur:
1. Download test datasets provided the authors of Ziyi et al.
    - https://sites.google.com/site/ziyishenmi/cvpr18_face_deblur
2. Make your test folder.
3. python test_face_deblur.py --dataroot ./facades/github/ --valDataroot <path_to_test_data> --netG ./pretrained_models/Deblur_epoch_Best.pth

