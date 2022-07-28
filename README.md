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
 
       @inproceedings{jacquelin:hal-03738153,
        TITLE = {{EFFICIENT ONE-SHOT SPORTS FIELD IMAGE REGISTRATION WITH ARBITRARY KEYPOINT SEGMENTATION}},
        AUTHOR = {Jacquelin, Nicolas and Duffner, Stefan and Vuillemot, Romain},
        URL = {https://hal.archives-ouvertes.fr/hal-03738153},
        BOOKTITLE = {{IEEE International Conference on Image Processing}},
        ADDRESS = {Bordeaux, France},
        YEAR = {2022},
        MONTH = Oct,
        KEYWORDS = {registration ; real-time ; sports ; dataset},
        PDF = {https://hal.archives-ouvertes.fr/hal-03738153/file/Efficient%20One-Shot%20Sports%20Field%20Image%20Registration%20with%20Arbitrary%20Keypoint%20Segmentation.pdf},
        HAL_ID = {hal-03738153},
        HAL_VERSION = {v1},
      }
