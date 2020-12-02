# Object detection project

The project's goal is to train and evaluate an SSD-resnet50 model to perform Object detection and create boxes around the three following classes:

* Vehicles
* Pedestrians
* Cyclists

The reason to perform object detection in self-driving cars is to locate possible obstacles and other agents such as the three listed above. This way we can avoid collisions and save lifes.

## EDA 

Location of Analysis: [Exploratory_Data_Analysis.ipynb](Exploratory_Data_Analysis.ipynb)

In the EDA we performed some visualizations of the dataset by looking at some images from Waymo's tfrecords and the GT boxes.

<p align="center"><img src=./Images/EDA_1.JPG height="500"/></p>

As you can tell from above, pink boxes represent Vehicles, cyan Pedestrians and gold Cyclists.

The sub dataset that we use from waymo's dataset comprised of 19802 images with 459264 objects in it. The distribution of the objects of interest is as follows:

<p align="center"><img src=./Images/EDA_2.JPG height="500"/></p>

As you can tell from above the distribution is umbalanced with the following percentages:

* Vehicles: 74%
* Pedestrians: 25%
* Cyclist: 1%

Therefore, this task was be very challenging to train as our SSD resnet was presented with more vehicle than pedestrians or cyclists samples.


### Cross validation

Location of Analysis: [Exploratory_Data_Analysis.ipynb](Exploratory_Data_Analysis.ipynb) and [create_splits.py](./create_splits.py)

Based on the findings in the EDA We decided to perform a cross validation with the following distribution:

* Training: 65%
* Validation: 15%
* Testing: 20%

To preserve the distribution an stratified cross-validation was performed. i.e. The images from all the tfrecords were mixed-up and pack together to preserve the 74% vehicles, 25% pedestrians, 1% Cyclist distributions along the 3 datasets. 

<p align="center"><img src=./Images/EDA_3.JPG height="400"/></p>

Finally, after we created the TF.Data.Datasets we write new tfrecords and distributed them in three different folders: train, val, test.

## Train and model evaluation

Location of Analysis: [Explore_augmentations.ipynb](Explore_augmentations.ipynb)

### Standard pipeline

The first configuration we tried with was the `pipeline.config` where only 2 augmentations were performed. 

* Random Horizontal flips 
    * probability: 0.5
* Random crop image
    * Min fraction of object covered: 0.0
    * Min aspect ratio(Aspect ratio bounds of cropped image): 0.75
    * Max aspect ratio(Aspect ratio bounds of cropped image): 3.0
    * Min area (ratio of cropped image to original image): 0.75
    * Max area (ratio of cropped image to original image): 1.0

After training the SSD-resnet on these conditions. The results of the tensorboard normalized total loss were as follows:

<p align="center"><img src=./Images/starter_pipeline_2.JPG height="300"/></p>

After the evaluation process we can also see the performance of the network as shown below:

<p align="center"><img src=./Images/starter_pipeline_1.JPG height="300"/></p>

You can see that the network finds out the vehicle objects mainly, This is expected as we saw in the EDA section where most of the data was from vehicle samples. However, it misclassifies some of them and the boxes are around certains features from the object and not the whole object.

### Augmented pipeline

To overcome the issues found in the previous pipeline we performed more augmentations as listed below:

* Random Horizontal flips 
    * probability of 0.6
* Random crop image
    * Min fraction of object covered: 0.0
    * Min aspect ratio(Aspect ratio bounds of cropped image): 0.75
    * Max aspect ratio(Aspect ratio bounds of cropped image): 3.0
    * Min area (ratio of cropped image to original image): 0.75
    * Max area (ratio of cropped image to original image): 1.0
* Random Adjust Brightness
    * max_delta (saturation from 0 to 1): 0.08
* Random Adjust Contrast (Randomly scales by a value between [min_delta, max_delta])
    * min delta: 0.8
    * max delta: 1.1
* Random Black Patches (Randomly adds black square patches to an image)
    * probability: 0.5
    * max black patches: 15

After training the SSD-resnet on these conditions. The results of the tensorboard normalized total loss were as follows:

<p align="center"><img src=./Images/Augmentation_2.JPG height="300"/></p>

As we can tell the loss decreases a bit more with a lower value of 0.66 whereas the previous config decreased only up to 1.6. 

Besides, we can share the remaining stats from tensorboard with this config as follows:

<p align="center"><img src=./Images/Augmentation_1.JPG height="300"/></p>

After the evaluation process we can also see the performance of the network as shown below:

<p align="center"><img src=./Images/Augmentation_3.JPG height="300"/></p>


You can tell that boxes are more accurate as they tend to soround the object and there does not seem to be misclassifications in some of the images(there is none in the image above). However, we still don't get objects coming from pedestrians and in the remaining images they are detected but with much less frequency. 

This same pattern can be observed in the [video](./animation.mp4) that we also display below as a GIF:


<div class="wrap">
    <img src="./Images/video.gif" />
    <br clear="all" />
</div>


To increase on accuracy we may require more augmentations and perhaps try to balanced the objects seen by the SSD model. Also it could be better to increase the size of the dataset by including more tfrecords from waymos google cloud buckets. 
