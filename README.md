# Fast Sport Fields Registration

## The Method

<p align="center">
  <img src="/images/pipeline.png" alt="Method's Pipeline" />
</p>

The model inputs an image, then detects arbitrary landmarks (not neccessary visual pattern recognizable by humans), then maps them with their position in the field. This mapping enables to use RANSAC algorithm to estimate the homography matrix.

As it is a one-shot method withou refinement, it is extremely fast (50 FPS on not so recent GPUs).

<p align="center">
  <img src="/images/race.gif" alt="original race">
  <img src="/images/homography.gif" alt="Homography Result" />
</p>


## RegiSwim Dataset
RegiSwim Dataset is available at [this link](https://drive.google.com/drive/u/0/folders/18BjEKYf5T2HYWi5k_rpXSmLtY92_md2g).



This dataset is a new benchmark for sport fields registration. It focuses on swimming pools, as these environments contain many interesting and unique properties to increase the challenge in image registration (levels of zoom, light saturation, changing background...).

![The Neptune Registration Dataset](/images/dataset.png)

 If you use it, please cite :
 
       @inproceedings{icip2022registration,
        author    = {Nicolas Jacquelin and Romain Vuillemot and Stefan Duffner},
        title     = {Efficient One-Shot Sports Field Image Registration with Arbitrary Keypoint Segmentation},
        booktitle = {IEEE International Conference on Image Processing},
        year      = {2022},
        url       = {⟨hal-03738153⟩}
      }
