#**Traffic Sign Recognition** 
###Self-Driving Car Engineer Nanodegree - _Project 2_
###By: **Soroush Arghavan**
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Capture.PNG "Visualization"
[image2]: ./Capture1.PNG "Distribution of Labels in the Training Data Set"
[image3]: ./Capture2.PNG "Distribution of Labels in the Validation Data Set"
[image4]: ./Capture3.PNG "Distribution of Labels in the Test Data Set"
[image5]: ./Capture4.PNG "PreProcessing"
[image6]: ./Capture5.PNG "5 New Test Images"
[image7]: ./Capture6.PNG "Feature Visualization of Convolution Layer 2"
[image8]: ./hard_images/0.jpg "Difficult Test 1"
[image9]: ./hard_images/1.jpg "Difficult Test 2"
[image10]: ./hard_images/2.jpg "Difficult Test 3"
[image11]: ./hard_images/3.jpg "Difficult Test 4"
---

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

Numpy was used to extract a summary of the data set characteristics as follows:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

First, let's take a look at the visual representation of the data sample. For this purpose, a random set of 10 images from the training data is displayed below.

![image1]

In order to better understand the distribution of the data, it is necessary to examine the labels and the number of samples associated with each one. The figure below portrays the distribution of labels among the training data set.

![image2]
![image3]
![image4]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

My first intuition about preprocessing the data was whether it would be better to process the images in greyscale or RGB. I assumed that RGB would be more beneficial since there will be more information layers to help classify the images. For example, detecting the color red in a sign would limit its categories to warning-type signs, detecting a high percentage of blue subpixels would mean certain permissions are allowed and so on.

I first ran the model without any preprocessing to see the result. However, I did not manage to achieve an acceptable accuracy. 

For the next trial, I added greyscale conversion as the first step of the preprocessing. Next, in order to minimize bias error and also to train the model with smaller weights, the images were normalized and then fed to the training pipeline. An example of processing stages on an image is shown below.

![image5]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used the provided training, validation and test data sets without alteration. The only step I took was to shuffle the data sets before feeding them to the training pipeline. One experiment was to flip the images along their vertical axes to create augmented data sets for training. However, this did not improve accuracy and is not discussed here.

Although, my experiment was not successful, it is possible that other techniques such as adding random noise to the training set, as well as adding distortion could improve the accuracy of the model.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for this step is contained in the fourth code cell of the IPython notebook. 
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 greyscale, normalized image			| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x6 	|
| RELU					|												|
| Max pooling 3x3	    | 1x1 stride, valid padding, outputs 30x30x6 	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 26x26x16 	|
| RELU					|												|
| Max pooling 2x2	    | 2x2 stride, valid padding, outputs 13x13x16 	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 9x9x24 	|
| RELU					|												|
| Max pooling 2x2	    | 2x2 stride, valid padding, outputs 4x4x24 	|
| Flatter				| 384        									|
| Fully connected		| outputs 240       							|
| Fully connected		| outputs 120        							|
| Fully connected		| outputs 43        							|
| Softmax				| cross entropy        							|
| Reduce mean			| loss calculation								|
| Optimizer				| Adam optimizer								|

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

I used the Adam optimizer for this project. After many iterations, it was found that a batch size of 60-90 would yield the best results for this model. Furthermore, some fluctuation in validation accuracy between different epochs were notices. In order to stabilize the gradients, the learning rate was reduced to 0.0007 and in return, the number of epochs was increased to 15 in order to counter the slow descent of the new learning rate. Furthermore, it was found that although the dropout probability did not show significant effect on the LeNet architecture, it was proven to be effective in increasing validation accuracy of the new architecture.

It was found that a keep probability of 0.6 yields optimized results for the model.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

The LeNet model was used as a stepping stone for this project. With a batch size of 128, learning rate of 0.001 and 10 epochs, the training data were fed directly to the model.

However, it was found that the resulting accuracy of the model on the verification data was arounnd 87 percent. In order to increase the accuracay of the model, and in order to reduce the possibility of overfitting, a dropout layer was added before the first fully connected layer with a dropout probability of 10 percent. Moreover batch size was reduced to 80 and the model was retrained.
Using the new parameters, the accuracy was increased to 91.7 percent. At this point, effects of the training parameters were examined in order to find the best approach to increasing the model accuracy. The results are shown below:

| Epochs	|	Learning Rate	|	Batch Size |	Keep Rate	| Accuracy 
|:---------:|:-----------------:|:------------:|:--------------:|:--------:| 
|10			|0.001		 		|100		   |0.9		 		|89.2|
|10		    |0.001 		  		|64 		   |0.9		 		|91.7|
|10			|0.001		  		|80			   |0.95            |91.8|
|15  		|0.001		  		|80			   |0.97	        |91.9|

The results showed no significant effect from any of the parameters. However, it was apparent that the learning rate could need to be decreased as there was noticeable fluctuation of validation accuracy between epochs.

Next, effect of generating more training samples was examined. In order to generate alternative training data, the training data were flipped along their vertical axes and added to the shuffled data set. This would introduce more training samples and counter the effects of unsymmetry in the images. However, the results did not show any promise as the validation rate remained below 92 percent.

At this stage, it was clear the the layers of the underlaying network are not perfectly suited for this data set. One mentionable approach was to adjust the size of the filters in the convolution layers. The size of the filters were altered between 3x3 to 5x5 and 7x7 pixels. The idea behind this approach was to try to force the layers to look for features of smaller or larger sizes. With no progress in sight, it was time to think of an alternative approach.

The LeNet model was developed for handwriting recognition on the MNIST data set which involves 10 classes. Our current data set has 43 different classes to identify with complicated and subtle differences between the images. Therefore, in order to be able to identify more complicated features, addition of another layer to the model could be of help.

In order to add another layer to the network, the parameters of the other two convolution layers such as filter sizes and depths, as well the the fully connected layer depths need to be readjusted. This is necessary in order to keep the difference between the depth of two adjascent layers to a low amount which could help in stability of the model.

My final model results were:
* training set accuracy of 98.1
* validation set accuracy of 93.7 
* test set accuracy of 91.6
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I searched for german traffic signs on the internet. I found one image that was worthy of testing which is image number 2. This image is interesting due to snow covering parts of the sign as well as the black and white background that adds to noise.

For other samples, I decided not to try stock images and instead, try taking samples from an actual dash cam footage of a vehicle in Munchen, Germany. For this purpose I used [this video](https://www.youtube.com/watch?v=Tq-Xziv-8xY). I took four interesting samples from this video and used them as test subjects. Two of these samples could be difficult to classify. Number 1 is slightly warped and number 4 has unfavorable lightling conditions.

![image6]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority Road     	| Priority Road 								| 
| Pedestrians     		| Speed limit (30km/h) 							|
| Speed limit (50km/h)	| Speed limit (50km/h)							|
| Speed limit (80km/h)  | Speed limit (80km/h)			 				|
| Ahead only			| Ahead only      								|

It was to my surprise that 4 of the images were classified correctly. The only failure was the sign that is covered with snow (number 2). This translates into an accuracy of 80 percent which is comparable to the test data set accuracy since the misclassified image was chosen as a difficult sample to identify and the sample size was extremely small. 

I also tried the following images on the model. These samples would be difficult to classify and were chosen for experimentation purposes. None of these test subjects were classified by the model correctly. My hypotheses would be the distortion, human face as well as having multiple signs in the image are the reasons for failure. Taking a deeper look at the features of each layer of the network could provide more insight into the features that caused the misclassification.

![image8] ![image9] ![image10] ![image11]

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The top 5 softmax possibilities for the 5 test images as well as the corresponding labels are as follows:

Image 0
Softmax 0
[  8.73608291e-01   7.64035806e-02   4.98321503e-02   7.26767830e-05
   3.14176978e-05]
Top Guesses 0
a) 5
b) 3
c) 2
d) 7
e) 8

Image 1
Softmax 1
[  9.99580681e-01   4.15015762e-04   4.09790073e-06   1.63666243e-07
   8.79331807e-08]
Top Guesses 1
a) 2
b) 5
c) 3
d) 1
e) 38

Image 2
Softmax 2
[  9.99913692e-01   4.60614065e-05   2.07322573e-05   1.52145349e-05
   1.48812865e-06]
Top Guesses 2
a) 12
b) 15
c) 40
d) 13
e) 1

Image 3
Softmax 3
[ 0.47948456  0.35532686  0.15199526  0.00506462  0.0026496 ]
Top Guesses 3
a) 37
b) 26
c) 18
d) 35
e) 24

Image 4
Softmax 4
[  9.99992251e-01   6.54967971e-06   3.21172536e-07   2.24598494e-07
   1.25790947e-07]
Top Guesses 4
a) 35
b) 34
c) 3
d) 40
e) 9
