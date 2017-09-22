# **Behavioral Cloning Project**

The project steps are the following:
* Data Collection
* Preprocessing of data
* Model Architecture
* Train & Test Model
* Conclusion

[//]: # (Image References)

[image1]: ./images/1.png "Image1 Before Preprocessing"
[image2]: ./images/2.png "Image1 After Preprocessing"
[image3]: ./images/1.png "Image2 Before Preprocessing"
[image4]: ./images/2.png "Image2 After Preprocessing"
[image5]: ./images/nVidia_model.png "NVIDIA CNN Model"

---

# **Data Collection**

In the project I have created my own training data, the data is for the following:
* Drive the car in a clockwise direction for one lap
* Drive the car in a counter-clock wise for one lap
* Drive the car from the lane to center for different lane form

I tried to use different data to generalize the model so I drived the car in a clockwise then counter clock wise then I have made several special training data for when the model is at the lane how to get out of it.

---

# **Data Preprocessing**

After collecting data I start my first step and that was to preprocess the data first I have cropped the image by **((60,20),(0,0))** I cropped the image for both  ```model.py ``` and ```drive.py``` , also I converted the colors from RGB to YUV after certain reference.
**Images below for before and after the preprocessing**

<img src="./images/1.png?raw=true" width="400px"> <img src="./images/2.png?raw=true" width="400px">
<img src="./images/3.png?raw=true" width="400px"> <img src="./images/4.png?raw=true" width="400px">
---

# **Model Architecture**

<img src="./images/nVidia_model.png?raw=true" width="400px">

NVIDIA Model is used it is composed of the following layers

| Layer         		|    
|:---------------------:|
| Convolution 5x5     	| 
| RELU					|
| Convolution 5x5     	| 
| RELU					|
| Convolution 5x5     	| 
| RELU					|
| Convolution 5x5     	| 
| RELU					|
| Convolution 5x5     	| 
| RELU					|
| Flatten layer	      |
| Fully connected		|
| Fully connected		|
| Fully connected		|
| Fully connected		|
| Dropout	(rate=0.5)| 
| Fully connected		|






