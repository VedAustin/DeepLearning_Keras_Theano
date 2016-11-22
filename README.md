# DeepLearning_Keras_Theano
Testing Deeplearning using Keras and Theano (via fine tuning or transfer learning) with Amazon Fashion catalogue images

Implemented the same Amazon fashion catalogue for image recogition this time using Keras, Theano and AWS powerful p2.xlarge instances.
The pre trained model used here is a 16 layer [`vgg16`] (http://www.robots.ox.ac.uk/~vgg/research/very_deep/) - one of the models that won Imagenet competitions a few years back (2014). Here we popped the last Dense layer from the original model (1000 categories) and inserted a new Dense layer (5 categories) and made only this layer trainable and every other layer `trainable=False`. To also speed up the fitting of this model, we precomputed the output from the convolution layers.

Accuracy wise this model performs poorly (possibly due to relatively high learning rate of 0.1) compared to tensorflow (not shown here) and **absolutely** requries GPUs unlike tensorflow which can comfortably run on cpus for this dataset. 

