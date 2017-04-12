#**Behavioral Cloning Project** 

---

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

At first I used the LeNet model. This worked okay, but didn't get the car to drive all the way around the track.

I then switched over to a model based on the NVidia end-to-end self driving model. [Self driving](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). This worked much better.

I gathered training data by driving around the track twice, backwards once, and recording reovery laps.

This worked well but didn't get past the bridge.


My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

