# Classifying Melanoma Tumor Images Using Convoluted Neural Networks

### Introduction

In 2021 alone, around 106,000 new melanomas will be diagnosed in the United States and around 7,180 Americans will die due to melanoma. The risk of developing melanoma increases over time, but it is one of the most commonly diagnosed cancers in young adults, particularly young women. Some of the common risk factors include, sun exposure, family history, fair skin, and the presence of moles. If skin cancer is diagnosed at an early stage survival rates are very high, but they rapidly drop once melanoma has spread from the initial site and advanced melanoma has a survival rate around 25% according to the American Cancer Society. 

![melanoma cell](https://user-images.githubusercontent.com/66225041/105279204-182f8200-5b75-11eb-8e7b-0826f7702f37.jpeg)       


Fig. 1 Image of melanoma cells.                   

![melanoma histo](https://user-images.githubusercontent.com/66225041/105279301-4614c680-5b75-11eb-9d85-b46c03572a6f.jpg)

Fig. 2 Melanoma histology. 

Machine learning has been used in medical imaging and diagnosis in the past few years, particularly in the field of radiology. Convolutional neural networks are one of the most promising tools. This type of algorithm takes an input image, assigns weights and biases to various features, and then classify the image. In the future this type of image analysis offers the promise of quickly and accurately diagnosing disease in patients at a lowered cost. 

 ![CNN image ](https://user-images.githubusercontent.com/66225041/105279379-66dd1c00-5b75-11eb-8ae1-45e7eb853069.jpeg)

Fig. 3 This image illustrates how convolutional neural networks analyze images in order to provide a classification prediction. 

### Data Cleaning and EDA 

Data was taken from Hospital Clínic de Barcelona, Medical University of Vienna, Memorial Sloan Kettering Cancer Center, Melanoma Institute Australia, The University of Queensland, and the University of Athens Medical School.

There were missing values from both the training and test data sets. These were imputed using a variety of criteria and methods. For missing age information, I looked at the mean and median ages, as well as the age distribution of the subjects. Since the distribution is normal, I imputed missing values as the median age, 50 years old. 

 
![age distribution](https://user-images.githubusercontent.com/66225041/105279423-7eb4a000-5b75-11eb-9b67-9ef234b0cfb7.png)
Fig. 4 Age distribution. 








I looked at the association between benign and malignant diagnosis in relation to the location and the two variables do not appear to be related. For this reason, I imputed the missing tumor location with the most common overall tumor location. 

![anatomy ](https://user-images.githubusercontent.com/66225041/105279461-90964300-5b75-11eb-92d9-5d3b15337b84.png) 
Fig. 5 The torso was the most common tumor location for both benign and malignant diagnoses. 

Similarly, I looked at the relation of sex to tumor diagnosis and there appeared to be no strong correlation. There were slightly more men than women in the study so missing sex values were imputed as ‘male’. 

At first glance, the only variable that appears to have bearing on diagnosis is age, with a malignant diagnosis tending to skew slightly older. 


 ![age distribution by target](https://user-images.githubusercontent.com/66225041/105279494-a99ef400-5b75-11eb-8695-be93e9f5dafc.png)
Fig. 6 Age distribution by target types 

One of the main issues with the dataset is that the classes are extremely unbalanced. There are far more images in the benign class than in the malignant class. In order to accurately fit a model to the data this issue will need to be addressed. There are a few ways to deal with unbalanced data. You can either oversample from the minority class, or you can undersample from the majority class. Undersampling can result in data loss however, so for this project I will be augmenting the data by performing transformations on the malignant images in order to generate enough images to sufficiently train the model. 

 ![benign_malignant](https://user-images.githubusercontent.com/66225041/105279620-f08ce980-5b75-11eb-936d-456666ca19c5.png)
Fig. 7 This image shows the imbalance in the two target data classes. 

 ![Undersampling_oversampling](https://user-images.githubusercontent.com/66225041/105279653-08646d80-5b76-11eb-9085-d0edb7dd64fd.png)
 
Some of the possible image transformations that can be performed are image blurring, changing the color channels for the images, randomly flipping the images vertically or horizontally, and changing the brightness of the images. For this particular project, I will be randomly flipping images vertically and horizontally, changing the hue, saturation, brightness, contrast and randomly rotating them. 

I examined a sample of 100 images from the dataset and found that there are a variety of shapes. The images will need to be resized to similar shapes in order for the model to fit the data accurately. I will be transforming all the images to be the same size and shape. 

 ![100 image shapes](https://user-images.githubusercontent.com/66225041/105279702-2631d280-5b76-11eb-9ce6-09d333fd55be.png)
Fig. 9 A sample of 100 images shows the range of image sizes in the data set. 

### Modeling 

#### ResNet-50 

ResNet-50 is a convolutional neural network with 50 layers. I am using a pre-trained version. This model won the ImageNet challenge in 2015 as it allowed successful training of very deep neural networks with many layers. 

 ![resnet502](https://user-images.githubusercontent.com/66225041/105279738-38ac0c00-5b76-11eb-920d-fd689661848e.png)

Fig. 10 A diagram showing how ResNet-50 works. 

For this model, the image data will be fit by the ResNet-50 neural network and the csv data will be fit by a feedforward neural network. The end result will be a prediction that classifies the images as either benign or malignant. 

#### EfficientNet 

This convolutional neural network uses a uniform compound scaling method for all dimensions rather than scaling dimensions arbitrarily. It was developed by Google and has been found to provide accurate results that require less memory than other common algorithms. 

![efficientnet2](https://user-images.githubusercontent.com/66225041/105279820-609b6f80-5b76-11eb-955f-9eb73899fead.png)
Fig. 11 A diagram showing how EfficientNet works.

 ![Efficient Net Diagram](https://user-images.githubusercontent.com/66225041/105279778-4a8daf00-5b76-11eb-86c0-dea9a485cc6c.png)
Fig. 12 EfficientNet was found to be much more accurate than other neural networks by Google. 

### Training 

The number of epochs for the training loop is set to 15. This is the number of passes through the entire training set the machine algorithm will be completing. Too many epochs can lead to overfitting of the training dataset, whereas too few may result in an underfit model. The patience variable is set to 3, referring to the number of epochs to wait before early stop if there is no progress on the validation set. Early stopping allows you to stop training once the model performance stops improving on a hold-out validation dataset. 

I implemented Test Time Augmentation into the training loop, the purpose of which is to perform random modifications to the test images. This is necessary since the classes are highly unbalanced as previously discussed.

The learning rate is set to 0.05. This hyperparameter controls how much to change the model in response to the estimated error each time the model weights are updated in each epoch. If the learning rate is too small there will be a long training process and if the learning rate is too large a suboptimal set of weights will be chosen for the model. 

I used K-fold cross validation and ROC to evaluate the performance of the models. K-fold cross validation is a resampling procedure that is used to estimate the skill of a machine learning model on unseen data. The ROC score is a curve that measures the probability of a binary outcome, used to evaluate classification problems.


 
![KFold](https://user-images.githubusercontent.com/66225041/105279850-727d1280-5b76-11eb-8f1b-804dec7324d3.png)
Fig. 13 How K-fold cross validation is implemented. 

### Model Evaluation


The best model was the B2 EfficientNet. The true negative value is very high, the model is good at identifying the benign images. While the false positive and false negative values are very low, the true positive value is also low. The model is not as good at identifying the malignant class of images. The out of fold ROC value was 0.976, which is a very good value. 

 
![Confusion Matrix](https://user-images.githubusercontent.com/66225041/105279872-8163c500-5b76-11eb-8c26-cb00cff46c4b.png)

The classification values look good for this model. The recall for the malignant category is the lowest value, while precision values for both classes are high. 

 ![Picture1](https://user-images.githubusercontent.com/66225041/105279936-a9ebbf00-5b76-11eb-940e-a8997c56a4a3.png)


### Conclusions

The model is skilled at classifying benign images but will need further tuning in order to classify a high percentage of malignant images correctly. A larger sample of actual malignant images might be a helpful tool to train the model further in the future. 

