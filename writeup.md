# **Behavioral Cloning Project**

The project steps are the following:
* Data Collection
* Preprocessing of data
* Model Architecture
* Train & Test Model
* Conclusion

---

# **Data Collection**

In the project I have created my own training data, the data is for the following:
* Drive the car in a clockwise direction for one lap
* Drive the car in a counter-clock wise for one lap
* Drive the car from the lane to center for different lane form

I tried to use different data to generalize the model so I drived the car in a clockwise then counter clock wise then I have made several special training data for when the model is at the lane how to get out of it.

---

# **Data Preprocessing**

After collecting data I start my first step and that was to preprocess the data first I have cropped the image by **((60,20),(0,0))** I cropped the image for both  ```sh model.py ``` and ```sh drive.py``` .
