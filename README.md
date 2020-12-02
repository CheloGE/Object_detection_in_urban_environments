# Object detection in an Urban Environment

## Data

For this project, we use data from the [Waymo Open dataset](https://waymo.com/open/). The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. 

Bear in mind that you will require to request access to get the buckets in the waymo google cloud.

## Structure

The data for this project is organized in the following way:

```
Images/
    - All images for the writeup and for any other illustration purposes
build/
    - In case you want to run the project in a local computer you should use Dockerfiles inside here
experiments/
    - exporter_main_v2.py: to create an inference model
    - model_main_tf2.py: to launch training and evaluation of your model
    - pretrained-models/: contains the checkpoints of the pretrained models. This folder should be downloaded using the following command:
        wget download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
        
- Exploratory_Data_Analysis.ipynb: In this file you can find all EDA performed such as distribution of the classes.
- Explore_augmentations.ipynb: In this file you can find the training and evaluation of the starter network, tryouts of different augmentation to increase performance and training and evalion steps from them.
- animation.mp4: A video of the network trained from one of the tfrecords
- animation_2.mp4: A second video of the network trained from another of the tfrecords.
- colab_setup.py: This file helps to setup the environment if you chose the docker option. However, this project was done entirely in google colab, so it is not needed.
- create_splits.py: As the name mentions, this file creates the splits between training, validation and testing.
- download_process.py: This file donwloads all data from Waymo's google cloud buckets
- edit_config.py: This file generates new config pipelines to train and evaluate our network
- filenames.txt: Waymo dataset is huge, thus this file lists only the tfrecords that we will use for the scope of this project.
- inference_videp.py: This file creates a video based on a SSD network and a tfrecord
- label_map.pbtxt: This file contains all class labels to map the numeric values from the boxes
- pipeline.config: This is the general pipeline that we use to add more augmentations and anyother changes such as the optimizer
- pipeline_augmentation.config: This file contains out attempt of increasing performance of pipeline.config file through data augmentation
- setup_google_colab.ipynb: This file is key as it contains all the dependencies for this project. This project was entirely developed inside google colab.
- utils.py: All helper functions for the other files above.
```

## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile in the [build directory](./build) or you can run the file `setup_google_colab.py` to create the environment in google colab.

* In case you choose the docker you can Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.
* For google colab you should copy the `setup_google_colab.py` and a zip of all the repo to the google colab folder and just call the `setup_google_colab.py` file to create all the required setup in one cell.

## Google colab setup

Add the following lines to build the environment for this project. (only works in google colab):
```
!python colab_setup.py
import os
os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/research/slim/'
os.environ['PYTHONPATH'] += ":/content/models"
```
**Note:** The `colab_setup.py` file is based on the following [reference](https://stackoverflow.com/questions/61934198/modulenotfounderror-no-module-named-nets-on-google-colab)


## Writeup

The writeup for this project can be found in the following markdown file
[writeup.md](./writeup.md). 



