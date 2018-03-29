
# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of neural network with 2 preprocesing layer followed by 5 convolutional layer followed by fully connected layers.
The model includes RELU layers to introduce nonlinearity (model.py lines 105-119). 

#### 2. Attempts to reduce overfitting in the model

Data augmentation is done by making left and right images as center camera images by changing steering angle by an offset.
Data point distribution:
![Hist_image](./dataPoints_histogram.jpg)

#### 3. Model parameter tuning

The model used an adam optimizer with learning rate of 0.0009 (model.py line 123).

#### 4. Appropriate training data

Training data provided by udacity with augmentation was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to get the vehicle driven on track and within drivable area.

My first step was to use a convolution neural network model similar to the NVidia arcitecture I thought this model might be appropriate because it was made to drive vehicle autonomously on different kind of tracks.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

Overfitting can be visualised while running the vehicle on track and most of the time the vehicle goes straight or left.

To combat the overfitting, I modified the model so that it can be trained for sharp left and right turns by using left and right camera images as recovery images.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like while taking left turn after bridge car was going straight off the track. To improve the driving behavior in these cases, I have tuned steering angle offset for left and right camera images and also to overcome the lag **Speed is been reduced to 4 in drive.py**.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes(model.py lines 101-119):
* **Cropping layer**        : to crop unwanted details from data
* **Lamda Layer**           : to preprocess(resize,normalize) the data
* **Convolution Layer**     : 24 5x5 kernels
* **Convolution Layer**     : 36 5x5 kernels
* **Convolution Layer**     : 48 5x5 kernels
* **Convolution Layer**     : 64 3x3 kernels
* **Convolution Layer**     : 64 3x3 kernels
* **Fully Connected Layer** : 5 fully connected layers to get steering angle as output


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I have used dateset provided by Udacity using center lane driving. Here is an example image of center lane driving:

![Center_Image](./originalImage.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from left or right side of lane. These images show what a recovery looks like starting from left and right to center :

Left Image :
![Left_Image](./Left.jpg)

Right Image :
![Right_Image](./Right.jpg)

Center Image :
![Center_Image](./Center.jpg)

Then I repeated this process on track where we there are sharp turns in order to get more data points.

To augment the data sat, I have used left and right images with angles thinking that this would help model learning the recovery path.For example, here is an image that has then been augumented as centre image:

![Left_Image](./Left.jpg)
 
![Right_Image](./Right.jpg)


After the collection process, I had 6429 number of data points. I then preprocessed this data by cropping and prprocessing layer of model.

Image after cropping :
![Cropped Image](./croppedImage.jpg)
Resized Image :
![Resized Image](./resizedImage.jpg)

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set help to determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by loss of around 0.022 I used an adam optimizer with learning rate 0.0009.
