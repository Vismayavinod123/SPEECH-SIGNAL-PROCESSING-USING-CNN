# SPEECH-SIGNAL-PROCESSING-USING-CNN

Introduction:
    The project target is two fold ; first is to identify the speech content irrespective of the speaker and second is to identify the speaker irrespective of the content. Unlike the state of the art feature extraction techniques, we use spectrograms of voice data as the input to CNN to build deep learning models. This has enhanced the learning process as none of the features were lost during tranformation and has improved the accuracy as data augmentation is easy in the case of images with built in libraries of keras. Free spoken digit dataset is used for training and validation. New voice samples and their spectrograms are created for testing the model. The results of training, validation and test are tabulated and plotted.
    
Package Used:   
        1. keras
        2. matplotlib.pyplot
        3. scipy
        4. os
        5. sys
        6. shutil
        
  
 Convolutional Neural Network
	A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms.
 1.4.1 Convolutional Layer
This layer performs the implicit feature extraction of images with the help of filters. Convolutional Neural Networks take advantage of the fact that the input consists of images and they constrain the architecture in a more sensible way.
We can compute the spatial size of the output volume as a function of the input volume size (W), the receptive field size of the Conv Layer neurons (F), the stride with which they are applied (S), and the amount of zero padding used (P) on the border. You can convince yourself that the correct formula for calculating how many neurons “fit” is given by (W−F+2P)/S+1. For example for a 7x7 input and a 3x3 filter with stride 1 and pad 0 we would get a 5x5 output. With stride 2 we would get a 3x3 output. 
ReLu Activation function
	ReLU is the abbreviation of rectified linear unit, which applies the non-saturating activation function  f(x)=max(0,x) . It effectively removes negative values from an activation map by setting them to zero. It increases the nonlinear properties of the decision function and of the overall network without affecting the receptive fields of the convolution layer. 
Other functions are also used to increase nonlinearity, for example the saturating hyperbolic tangent   f(x)=tanh(x),f(x)=|tanh(x)| , and the sigmoid function  . ReLU is often preferred to other functions because it trains the neural network several times faster without a significant penalty to generalization accuracy.
Pooling Layer
	The Pooling layer is responsible for reducing the spatial size of the Convolved Feature. This is to decrease the computational power required to process the datathrough dimensionality reduction. Furthermore, it is useful for extracting dominantfeatures which are rotational and positional invariant, thus maintaining the process of effectively training of the model.
There are two types of Pooling: Max Pooling and Average Pooling. Max Pooling returns the maximum value from the portion of the image covered by the Kernel. On the other hand, AveragePoolingreturns the average of all the valuesfrom the portion of the image covered by the Kernel.
Dropout Layer
Dropout is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass. If neurons are randomly dropped out of the network during training, that other neurons will have to step in and handle the representation required to make predictions for the missing neurons. This is believed to result in multiple independent internal representations being learned by the network.
Fully Connected Layer
Finally, after several convolutional and max pooling layers, the high-level reasoning in the neural network is done via fully connected layers. Neurons in a fully connected layer have connections to all activations in the previous layer, as seen in regular (non-convolutional) artificial neural networks. Their activations can thus be computed as an affine transformation, with matrix multiplication followed by a bias offset (vector addition of a learned or fixed bias term).



DATASET DESCRIPTION AND OBJECTIVES

Dataset used in this project for training and validation of the CNN model is spoken digit dataset that is freely available on internet. It comprises of audio recordings from 4 different speakers. There are 500 recordings of each speaker with 50 recordings of each digit (0-9). Test Dataset is developed from the original voice samples of the some peoples

        
        
Result:

        The aim of this project has been to explore application of Convolutional Neural Networks (CNNs) for speech signal recognition. CNNs are widely used in Computer Visison applications. The catch was to convert audio signals into spectrogramas, as discussed in previous chapters, to obtain information content of the signal in grey scale variations format. 
        
        
Conclusion:

        Deep learning has invaded into all facets of life with incredible impacts. The capability to process unstructured data and to solve complex problems make it the technology of the present and future. This project analyzes the effect of using CNN for speech recognition and speaker identification. From the results it is obvious that CNN based speech enables systems can be more accurate, fast and does not require  feature extraction. The prediction results of the test data are not highly promising but can be further improved by adjusting the sampling rate and by training the model with female voice recordings.


Model Summary:
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 64, 64, 32)        896       
_________________________________________________________________
batch_normalization_1 (Batch (None, 64, 64, 32)        128       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 63, 63, 48)        6192      
_________________________________________________________________
batch_normalization_2 (Batch (None, 63, 63, 48)        192       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 62, 62, 120)       23160     
_________________________________________________________________
batch_normalization_3 (Batch (None, 62, 62, 120)       480       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 31, 31, 120)       0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 31, 31, 120)       0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 115320)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               14761088  
_________________________________________________________________
batch_normalization_4 (Batch (None, 128)               512       
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 64)                8256      
_________________________________________________________________
batch_normalization_5 (Batch (None, 64)                256       
_________________________________________________________________
dropout_3 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                650       
=================================================================
Total params: 14,801,810
Trainable params: 14,801,026
Non-trainable params: 784

![image](https://user-images.githubusercontent.com/62199904/156126258-3210d0d2-bd58-4af2-91ac-06485ea8c31f.png)


