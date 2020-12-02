# Object detection project

The project's goal is to train and evaluate an SSD-resnet50 model to perform Object detection and create boxes around the three following classes:

* Vehicles
* Pedestrians
* Cyclists

The reason to perform object detection in self-driving cars is to locate possible obstacles and other agents such as the three listed above. This way we can avoid collisions and save lifes.

## EDA

In the EDA we performed some visualizations of the dataset by looking at some images from Waymo's tfrecords and the GT boxes.

<p align="center"><img src=./Images/EDA_1.JPG height="500"/></p>

As you can tell from above. pink boxes represent Vehicles, cyan Pedestrians and gold Cyclists.

Another task is that 

```
## Submission Template

### Dataset
#### Dataset analysis
This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.
#### Cross validation
This section should detail the cross validation strategy and justify your approach.

### Training 
#### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

#### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.

```
