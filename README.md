# DeepLearning_Keras_Theano
Testing Deeplearning using Keras and Theano (via fine tuning or transfer learning) with Amazon Fashion catalogue images

Implemented the same Amazon fashion tech catalogue for image recogition this time using Keras, Theano and AWS powerful p2.xlarge instances.
The pre trained model used here is a 16 layer [`vgg16`] (http://www.robots.ox.ac.uk/~vgg/research/very_deep/) - one of the models that won Imagenet competitions a few years back (2012?).
Accuracy wise this model performs poorly compared to tensorflow and **absolutely** requries GPUs unlike tensorflow which can comfortably run on cpus for this dataset.

