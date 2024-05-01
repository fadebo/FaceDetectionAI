1	Introduction
Neural Network has been a fascinating and intriguing topic to me since I learned about them, neural network has a variety of applications, from stock market prediction to aerospace the human brain’s replica (interconnected layers) can do many things with some minimal contributions. There are two ways in which a neural network can be approached using python. The first method is to create from scratch, this method can be considered good because it gives a solid foundation when it comes to creating neural networks. The second method is by using neural network libraries such as Keras, Kivy, and TensorFlow, as it is more advanced and the fastest way to create a neural network. In this project, we take the development approach and use a generic object-detection pipeline to create a face-detection model and enable the AI to get images through our webcam. Specifically, the object detection model is split into two different models: A classification model which we are going to use to classify what the type of object is and in this project our object is one class which is ‘face’. The other is a Regression model which we would use to estimate the coordinates of the bounding box, the coordinates needed are basically two the top left or top right, and the bottom left or bottom right.

2	Background/Literature Review
Face detection is an artificial intelligence (AI) based computer technology that is used to detect and identify faces in images. Face detection is mostly used for security and entertainment. Face detection has developed from machine learning(ML) to a highly developed artificial neural network (ANN); Face detection plays a major role in facial recognition. 
2.1.  Facial recognition
In face recognition without face detection, it is impossible to allow the camera to detect and locate an image of a face, that may be alone or with many people. Face recognition is a category of biometric security, and face recognition is also used to unlock some devices either phones or PCs. Typically, facial recognition does not rely on a massive database of photos to determine someone’s identity and recognize them.

2.1.a  Uses of Facial recognition
There are various ways face recognition has been involved in our lives, some are: 
Unlocking phones or PCs: Most recent phones, use face recognition to unlock devices, face recognition helps to protect the personal information of the owner of the device and without the owner unlocking the device it is almost impossible for someone else to open that device using their face. 
Law enforcement: Face recognition is used by law enforcement to detect an offender and compare them to their face recognition database.
 Airports and border control: Face recognition are used in airports when travellers hold biometric passports, it helps in improving the security of the airport and makes travelling easier by skipping the long lines and going to automated ePassport control.
 There are also more uses of facial recognition such as: finding missing persons, reducing retail crime, improving retail experiences, and banking.
2.2 How does Facial detection work? 
	Facial detection technology uses algorithms to detect whether images include a face. Face detection software mainly relies on smart algorithms that biometrical map out facial characteristics acquired in pictures and video frames. Face detection applications use algorithms and machine learning(ML) to find faces within larger images, which often incorporate other non-face objects such as the background and other body parts. Face detection algorithms start off by detecting the eyes then the eyebrows and other parts of the face, when the algorithm finishes detecting everything in the face region it applies a test. 
To improve the accuracy of the face detection application the algorithms need to be trained on large datasets incorporating thousands if not a million positive and negative images. This training will help the algorithm to be able to detect if there is a face in an image or not and where they are located.
 2.2.a Software used to make face detection applications:
	These methods more or less use the same process for detecting faces
•	Neural Networks: This is also known as artificial neural networks(ANNs) or simulated neural networks(SNNs)
 ![image](https://github.com/fadebo/FaceDetectionAI/assets/168660530/cf82bea7-58ec-4112-b235-4be6a2d76c5e)

•	OpenCV(Open Source Computer Vision Library) is an open-source computer vision and machine learning software library
 ![image](https://github.com/fadebo/FaceDetectionAI/assets/168660530/5dd837f8-02ac-4df9-bce8-3d5b601c99ee)

•	MATLAB
 ![image](https://github.com/fadebo/FaceDetectionAI/assets/168660530/596c2341-0bf5-4e8c-a6bb-557c44f3aac8)

•	TensorFlow helps implement best practices for data automation, model tracking, performance monitoring, and model retraining.
 ![image](https://github.com/fadebo/FaceDetectionAI/assets/168660530/5add8dd0-a01c-43a7-9289-035e2d10f57b)

The methods used in face detection can be rule-based, feature-based, template matching or appearance-based. 
•	Rule-based methods: These methods describe the face based on the rules defined.
•	Feature-based methods: These methods use features such as eyes, nose, and mouth to detect a face.
•	Template-based methods: These methods compare the images based on a standard face pattern to detect a face.
•	Appearance-based: These methods use statistical and machine learning to detect a face.
In face detection detecting faces in images can be problematic due to a variety of factors, such as orientation, expression etc. In modern years this problem has worked on and face detection using deep learning offers the advantage of substantially surpassing former computer vision methods.
A convolutional neural network(CNN) is a type of artificial neural network used in image recognition and processing that is specifically designed to process pixel data.
2.3 Advantages of face detection
There are various advantages of face detection such as:
•	Improved security: Face detection improves security for devices such as phones and makes it harder for hackers to steal or change anything.
•	Easy to integrate: Face detection and face recognition are easy to integrate.
•	Automated identification: Face detection allows the identification process to be automated, which saves time while increasing accuracy.
2.4 Disadvantages of face detection
•	Enormous data storage burden: Face detection technologies require enormous data storage.
•	Vulnerability: While face detection provides automated identification it can easily be hurled off by changes in appearance.
3	Dataset
TensorFlow, JSON, NumPy, matplotlib and pyplot.  TensorFlow is used in this project for implementing model tracking and model retraining. JSON (JavaScript Object Notation) is used in this project for storing temporary data. NumPy in this project is used when working with arrays 

4	Results
Face detection is achieved by detecting elongated regions in the skin map by properly modifying the Orientation Matching(OM) technique. This technique detects circular objects of radius in the interval [rm, rM], computing the orientation matching transform of the input picture and taking the heights of the transform, which relate to the centres of the circular patterns.

5	Discussion
Week 1 Progress
	The project that I am working on is “Face detection”, in the project proposal I have given a brief introduction of the project. For this week’s progress, I will be doing the following: 
1.	Installing and setting up the dependencies needed
a.	I have six(6) different dependencies that I’m going to use in this project they are:
•	LabelMe: This dependency allows me to do a lot of different types of annotations
•	TensorFlow & TensorFlow-GPU: These dependencies are going to assist me with deep learning
•	OpenCV-python: This dependency is going to be used for real-time detection and capturing images
•	Matplotlib: This dependency is going to be used for the rendering 
•	Albumentations: This dependency is going to be used to do the data augmentation.
The dependencies listed above are the dependencies needed to be installed in the python environment. This took me some time because I haven’t worked with most of the dependencies before, so after setting this up we move to the next step which is
2.	 Getting the images
a.	So, in order the get the images I am going to be using OpenCV, I am  going to import four different libraries which are:
•	OS: This library makes it easier for me to navigate through different file paths or join different file paths together. “import os”
•	 Time: This library helps me when I am collecting images, I am giving myself a little bit of time to move around. “import time”
•	UUID: This library helps me to create a unique uniform identifier name for my images. ”import UUID”
•	CV2(Computer vision): This library helps me work with different cameras and sensors

Now that the four libraries are now imported, I moved to the next step which is :
b.	Defining paths: So basically, now that I am done with importing the libraries, It’s time for me to define the paths that I need for this project, and they are:
•	Defining the location of the images: This is the location where I placed my images 
•	I created a folder named source and, in that folder, I created another folder named “images” and “labels”
•	defining the number of images: This is the number of datasets that I collected, I first used 15 but because I want as much variability, I decided to go with 25, if it’s still not enough I will increase it by 25 another 3 times. So, I might be expecting 100 different datasets
c.	Setup connection with webcam
•	Establishing a connection with my webcam
•	Loop through the images
•	Print out the image currently collecting
•	Read from the capture device
•	Capture the frame and write it down using CV2
•	Setting the unique filename for the images using UUID
•	Use the time library and sleep for 0.6 a second between each frame
•	Create the break codes that will allow me to break out of the loop 
d.	Annotate the images using label me
3.	This will be all for this week, still got a lot to do, the code is attached to the file.
Week 2 Progress 
	This is the continuation of the “Face Detection” software. For this week’s progress, I will be doing the following:
1.	 Creating a function for loading images
a.	So, to create a function for loading images I must import a few key dependencies
•	TensorFlow: This dependency will help me to build my data pipeline and for building my deep learning model.
•	JSON: This dependency is going to be used to load my labels into my python pipeline.
•	NumPy: This dependency is going to be used to help with data pre-processing.
•	Matplotlib: This dependency is going to be used for the visualisation of images.
b.	Limit the GPU Memory growth by default 
•	So, to start this step I had to first configure my GPU, this took me a while.
•	So, I stopped TensorFlow from taking all the VRAM on my GPU. It reduces the out of memory errors
•	Check if the GPU is available
c.	Load images into the TensorFlow data pipeline
•	Perform a wildcard search for (.jpg) in the images folder
•	Create a load image function to load an image for object detection, read the file path to the byte-encoded image and decode the images to a uint8 tensor
•	I applied the function on each value in the dataset using the map component
array([[[107, 98, 93], [106, 97, 90], [106, 97, 90], ..., [100, 117, 111], [102, 117, 112], [100, 115, 110]], [[108, 97, 91], [106, 97, 88], [107, 97, 87], ..., [ 98, 115, 109], [101, 116, 111], [101, 116, 111]], [[110, 98, 86], [107, 98, 83], [109, 97, 83], ..., [100, 115, 110], [100, 115, 108], [ 99, 114, 107]], ...,
...
[147, 111, 95], ..., [154, 174, 175], [155, 173, 173], [155, 173, 173]]], dtype=uint8)
The above are some of the mapped datasets
d.	Visualize the images
•	Batch the images up using the TensorFlow dataset API
•	Use matplotlib and subplots class to loop through and visualize the images.
 ![image](https://github.com/fadebo/FaceDetectionAI/assets/168660530/6e4df2b4-c9c2-4c18-9bcc-dbc1e3929bc5)
![image](https://github.com/fadebo/FaceDetectionAI/assets/168660530/69bb18b8-c105-4db0-b3d1-8101274771b4)

 
2.	So, after creating a function for loading and visualizing my images I used the train/test split and validate data
a.	I created a Train, Test and Val folder
b.	I created the labels and images folder inside the Train, Test and Val folder
c.	I’m still working on a code for the train, test and Val code, so for now I have done it manually should have the code by next week’s progress submission(So I had 50 images, split 35 for the train, 10 for the test and 5 for the Val)
3.	Move the labels for the images that have been split from the root folder to the train, test and Val folders
a.	Loop through the Train, Test and Val folders and move the associated labels for the images in the image folder of  the Train, Test and Val folders to the labels folders of the Train, Test and Val folders
Week 3 Progress 
	This is the continuation of the “Face Detection” software. For this week’s progress, I will be doing the following:
1.	So, as I said in my last progress, I would create a code for the train/Val/test, and I did that using the split folders package. I was kind of stuck on which package I should use that can simplify my code without affecting the speed of my program and I eventually choose the split folders package
a.	So, I had 50 images, split 35 for the train, 10 for the test and 5 for the Val
b.	So, another reason I chose split folders is that it works with any file types
2.	Setting up Augmentation for labels and images
a.	So, Augmentation is one of the critical steps  that are needed for this project
b.	The first thing I did was to import my library which I’ll be using, and which is albumentation
•	Albumentation is amazing when setting up my bounding box, it makes it easier and allows me to skip some excess codes.

c.	Defining my augmentation pipeline: I have six different argumentations that I am going to apply for my program which are:
•	Random Crop: This is going to be used to specify how big, augmented images are going to be.
•	Horizontal Flip
•	Vertical Flip
•	Random Brightness Contrast
•	Random Gamma
•	RGB Shift
d.	Load a test image and labels from the train folder
e.	Rescale coordinates to match image resolution 
f.	Applying Augmentations
•	So, I visualized the image and set the true tuples that I need to draw the bounding box
•	Rescale it to represent the size of the image because I need to transform it into normalized values
•	Untransformed it to go and render, otherwise it’s going to look small
•	So, I represented it as an integer and passed it on as a tuple because that’s what OpenCV expects
•	Specifying the colour of the bounding box to be green in bgr format
•	 ![image](https://github.com/fadebo/FaceDetectionAI/assets/168660530/42d7350b-8902-42e4-a586-0d1baf8efe82)

•	 ![image](https://github.com/fadebo/FaceDetectionAI/assets/168660530/35258aa3-3d87-468d-b445-748ad8c74a8d)


3.	Create a folder called data and create train/test/Val subfolders and in those subfolders create an image and labels folder
4.	Run the Augmentation pipeline over the train/ Val/ test partitions for all images
a.	Loop through the train, test, and Val folders
b.	Collect all images and double-check if an annotation exists for that image
c.	If an annotation doesn’t exist, I set up a default coordinate and if it does exist, I extracted the coordinates and rescaled it to represent the size of the image because I need to transform it into normalized values
d.	Next, I’m going to create 50 images augmented images for every single base image
e.	That means for the 50 images that I previously created; I’m going to be multiplying by 50 augmented images that I’m going to be able to use
f.	Run the images through the augmentation pipeline 50 times to create 50 different images
g.	Write out the augmented image and place it inside a folder called data
h.	Transform the coordinates and write down the annotations using JSON dump

Finalizing project:
1.	loading images into my TensorFlow dataset
2.	Create a load label function to load a label for object detection, read the file path to the byte-encoded label and decode the label to a uint8 tensor
3.	loading labels into my TensorFlow dataset
4.	checking the lengths of the image and labels partition
a.	(35, 35, 10, 10, 5, 5)
5.	combining all examples of the datasets
(array([[1],
        [0],
        [1],
        [0]], dtype=uint8),
 array([[0.5054, 0.477 , 0.788 , 0.998 ],
        [0.    , 0.    , 0.    , 0.    ],
        [0.5054, 0.477 , 0.788 , 0.998 ],
        [0.    , 0.    , 0.    , 0.    ]], dtype=float16))
6.	Use matplotlib and subplots class to loop through and visualize the sample data
a.	![image](https://github.com/fadebo/FaceDetectionAI/assets/168660530/90e7b293-c6f7-4da0-a662-989977cfe566)

b.	  ![image](https://github.com/fadebo/FaceDetectionAI/assets/168660530/3882fe91-3313-4e54-ba48-1be9ad553234)

7.	Import our base model class
8.	import some TensorFlow layers such as (Input, Conv2D, Dense, and GlobalMaxPooling2D)
9.	import the neural network VGG16 
a.	it’s mainly used for image classification 
10.	Deep learning model
a.	Object detection has two parts which are: classification and regression 
b.	Use the VGG16 base architecture and add on the final prediction layers one for classification and the other for regression
i.	Create an instance of VGG
ii.	show what the neural network looks like
c.	Building the neural network - Use the function API for the model that will allow me to have two different loss functions and combine them, one for the classification and the other regression model
i.	Specify the input layer, using the Input class that has the function API, so it makes things easier. Set the shape to 120 pixels by 120 pixels by 3 pixels
ii.	Take the input layer and pass it through the VGG16 layer
iii.	Setup the two prediction heads respectively the classification and regression models
•	Condense all the information from the VGG layer using the GlobalMaxPooling2D layer
•	Create a dense layer for the classification model which has 2048 units with activation of ‘relu’
•	The value is then passed to another classification model dense layer which has one output and has activation of ‘sigmoid’
•	Create a regression model dense layer which has 2048 units with activation of ‘relu’
•	The value is then passed to a regression model dense layer which has four outputs and has an activation of ‘sigmoid’
iv.	Combine it all together using the Model API, we pass through our inputs and outputs
11.	testing the built network
a.	Create an instance of the built model function
b.	Check the summary of the neural network
c.	Grab a sample out of the training pipeline
d.	Pass through the images to the tracking face model
12.	Term my optimizers and losses
a.	specifying my learning rate decay
b.	setting up the optimizer using the Adam optimizer
c.	create the localization loss
i.	Getting the actual and predicted width & height
ii.	Calculating the delta coordinates and size
d.	Pass the classification loss to the training pipeline
e.	Pass the regression loss to the localization loss
f.	Testing the losses
13.	Training the Neural network
a.	Creating a model class
i.	Create an init method to pass through initial parameters
ii.	Create a compile method and pass through the losses and optimizer
iii.	Create a train step where the neural network training takes place
iv.	Create a test step it’s triggered by the validation dataset and it’s almost identical to the train step the only difference is that I’m not applying back crop and calculating the gradient
v.	Create the call method
b.	Subclass the model
14.	Specify the log directory where the tensor board model will log out to
15.	Create a tensor board call back
16.	Save the training history
17.	Use matplotlib and subplots class to loop through and visualize the Loss
a.	 ![image](https://github.com/fadebo/FaceDetectionAI/assets/168660530/99b70061-e63d-436b-b2a5-2d971d59a00a)

18.	Making predictions
19.	Use matplotlib and subplots class to loop through and visualize the predictions
a.	 ![image](https://github.com/fadebo/FaceDetectionAI/assets/168660530/1f66d6a1-afb7-4510-bbd4-d441413237a0)

20.	Saving & loading model
21.	Getting Real-Time Detection by controlling the main, label-rectangle and moving with the face detected
a.	Capturing the video frame
b.	cutting down the pixels
c.	converting to RGB
d.	reseize it to be 120 x 120 pixels
e.	divide by 255 to scale it down
f.	perform rendering

6	References
https://www.kaspersky.com/resource-center/definitions/what-is-facial-recognition
https://www.techtarget.com/searchenterpriseai/definition/face-detection	
https://www.cameralyze.co/blog/5-reasons-why-face-detection-is-important
https://www.tensorflow.org/learn
https://opencv.org/about/
https://www.sciencedirect.com/topics/computer-science/face-detection
https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch
https://github.com/wkentaro/labelme

