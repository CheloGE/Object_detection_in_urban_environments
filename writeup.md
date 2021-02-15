# Object detection project

* **Author:** Marcelo Roger García
* **Repo link:** https://github.com/CheloGE/Object_detection_in_urban_environments


The project's goal is to train and evaluate an SSD-resnet50 model to perform Object detection and create boxes around the three following classes:

* Vehicles
* Pedestrians
* Cyclists

The reason to perform object detection in self-driving cars is to locate possible obstacles and other agents such as the three listed above. This way I can avoid collisions and save lifes.

## EDA 

Location of Analysis: [Exploratory_Data_Analysis.ipynb](Exploratory_Data_Analysis.ipynb)

In the EDA I performed some visualizations of the dataset by looking at some images from Waymo's tfrecords and the GT boxes.

<p align="center"><img src=./Images/EDA_1.JPG height="500"/></p>

As you can tell from above, pink boxes represent Vehicles, cyan Pedestrians and gold Cyclists.

The sub dataset that I use from waymo's dataset comprised of 19802 images with 459264 objects in it. The distribution of the objects of interest is as follows:

<p align="center"><img src=./Images/EDA_2.JPG height="500"/></p>

As you can tell from above the distribution is umbalanced with the following percentages:

* Vehicles: 74%
* Pedestrians: 25%
* Cyclist: 1%

Therefore, this task was be very challenging to train as our SSD resnet was presented with more vehicle than pedestrians or cyclists samples.


### Cross validation

Location of Analysis: [Exploratory_Data_Analysis.ipynb](Exploratory_Data_Analysis.ipynb) and [create_splits.py](./create_splits.py)

Based on the findings in the EDA We decided to perform a cross validation with the following distribution:

* Training: 75%
* Validation: 15%
* Testing: 10%

To check the performance of our model a simple cross validation approach was implemented. The validation is used to check the generalization of the data while training. Whereas, the testing set is completely used for performing the inference section and creating videos to visualize how the model generalized on completely new data. 

<p align="center"><img src=./Images/EDA_4.JPG height="400"/></p>

Finally, after I created the TF.Data.Datasets I write new tfrecords and distributed them in three different folders: train, val, test.

## Train and model evaluation

Location of Analysis: [Explore_augmentations.ipynb](Explore_augmentations.ipynb)

### Standard pipeline

The first configuration I tried with was the `pipeline.config` where only 2 augmentations were performed. 

* Random Horizontal flips 
    * probability: 0.5
* Random crop image
    * Min fraction of object covered: 0.0
    * Min aspect ratio(Aspect ratio bounds of cropped image): 0.75
    * Max aspect ratio(Aspect ratio bounds of cropped image): 3.0
    * Min area (ratio of cropped image to original image): 0.75
    * Max area (ratio of cropped image to original image): 1.0

After training the SSD-resnet on these conditions. The results of the tensorboard normalized total loss for the training loss decreased quickly. However, the validation didn't even show in the graph as it didn't decrease, as shown below:

<p align="center"><img src=./Images/starter_pipeline_2.JPG height="300"/></p>

However, I got the following performance from the evaluation process: 

<p align="center"><img src=./Images/starter_pipeline_1.JPG height="300"/></p>

You can see that the network finds out the vehicle objects mainly, This is expected as I saw in the EDA section where most of the data was from vehicle samples. However, it misclassifies some of them and the boxes are around certains features from the object and not the whole object. No pedestrians are found. 

## Augmented pipeline

To overcome the issues found in the previous pipeline I performed more augmentations as listed below:

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

An image of the exploratory of data augmentation section is shown below:

<p align="center"><img src=./Images/EDA_3.JPG height="300"/></p>

For instance, in the figure above, the black boxes try to add a certain level of occlusion during the training process.


After training the SSD-resnet on these conditions. The results of the loss in tensorboard were as follows:

<p align="center"><img src=./Images/object_detection2.JPG height="300"/></p>

As I can tell the validation loss now decreases in pair to the training loss.

Besides, I can share the remaining stats from tensorboard related to the mAP as shown below:

<p align="center"><img src=./Images/object_detection1.JPG height="300"/></p>

We can see from the image above how the mAP has increased slowly with each step. I selected a very low learning rate to guarantee improvement in performance. The downside is that I require a lot more time to train. Unfortunately, since this project was developed in google colab I had a limited time frame to train the model as the training can be idle for a couple of hours only. 

On the other hand, the learning rate approach was to get a warm up value of 0.00133 and slowly increase it up to 0.004 in the firts 2000 steps to finally decrease it very slowly upto the 10000 steps, as shownb below:

<p align="center"><img src=./Images/object_detection3.JPG height="150"/></p>

After the evaluation process I can also see the performance of the network as shown below:

<p align="center"><img src=./Images/object_detection4.JPG height="300"/></p>


You can tell that boxes are more accurate as they tend to soround the object and there does not seem to be misclassifications in some of the images(there is none in the image above). However, I still don't get objects coming from pedestrians and in the remaining images they are detected but with much less frequency. 

This same pattern can be observed in the wideos shown below taken from the testing set. They are displayed in a GIF format:


<div class="wrap">
    <img src="./Images/video1.gif" />
    <br clear="all" />
</div>

<div class="wrap">
    <img src="./Images/video2.gif" />
    <br clear="all" />
</div>

<div class="wrap">
    <img src="./Images/video3.gif" />
    <br clear="all" />
</div>

<div class="wrap">
    <img src="./Images/video4.gif" />
    <br clear="all" />
</div>

<div class="wrap">
    <img src="./Images/video5.gif" />
    <br clear="all" />
</div>

<div class="wrap">
    <img src="./Images/video6.gif" />
    <br clear="all" />
</div>


To increase on accuracy I may require more augmentations and perhaps try to balanced the objects seen by the SSD model. Also it could be better to increase the size of the dataset by including more tfrecords from waymos google cloud buckets. Fianlly, a resnet 101 or 150 instead of the Resnet50 can be used to get better results. 

Location main video: [here](./animation.mp4)
