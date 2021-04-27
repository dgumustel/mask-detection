# Face Mask-Detection: Multiclass Classification using Convolutional Neural Networks (CNN)

Team members: Umair Evans, Derya Gumustel, Tarun Nappoly & Amir Semsarzadeh

COVID-19 is a global pandemic that has expanded the requirement for masks in public areas as means of protecting ourselves and curtailing the virus. Masks are not only crucial, but also need to be worn and worn correctly, to be effective. What may appear as a no-brainer has become a difficult and cumbersome challenge to enforce both on a public and private level. 

We initially observe that object detection through CNN is an effective strategy in determining
whether an individual is wearing a mask properly, improperly, or not at all. We define correctly as having nose, mouth, and chin covered. Improperly indicates that one of these three areas of the face are exposed or partially exposed.

Our goal is the application of this model to both photos and video to enhance and improve public well being.



### Table of Contents

* [Problem Statement](#user-content-problem-statement)
* [Executive Summary](#user-content-executive-summary)
* [Conclusions and Recommendations](#user-content-conclusions-and-recommendations)
* [Additional Information](#user-content-additional-information)

---

### Problem Statement

Can our model reliably predict who is wearing a mask properly, improperly, or not wearing one at all?

---

### Executive Summary
The essential premise of our experiment is to use machine learning to create a distinction between three classifications. These would serve as a means to support a strategic integration with both camera system working in stills as well as in motion to diagnose our particular classes.

#### Data

Data provided by Tero Karras who is a Principal Research Scientist at NVIDIA.

[Face dataset without masks](https://github.com/NVlabs/ffhq-dataset)

[Face dataset with masks](https://github.com/cabani/MaskedFace-Net)


The dataset consists of 70,000 high-quality PNG images at 1024×1024 resolution and contains considerable variation in terms of age, ethnicity and image background. It also has good coverage of accessories such as eyeglasses, sunglasses, hats, etc. The images were crawled from Flickr and are inherently subject to the bias and disposition of that network.

For our purposes we took a small sample (started with 3,000 images before scaling up to 9,000) and reduced the images to 256 by 256 pixels in an attempt to improve computational efficiency. Various automatic filters were used to prune the images to optimize learning. To deal with the constraints of computational resources on typical machines we reduced the images outside of the modeling process. We must also stress that our data as it relates to one of our classifications might be imbalanced, and future iterations should strengthen the incorrectly masked class.

It is important to note that while we may have no lack of diversity as it relates to people, it is a different story when it comes to masks. The blue surgical mask that we often see is the most common, but future iterations of this experiment should contain a larger diversity of masks.

The images are contained in the following folders: 

Folders 04000, 05000, and 06000 contain photos of people wearing masks correctly.

Folders 09000, 10000, and 11000 contain photos of people not wearing masks.

Folders 13000, 14000, and 15000 contain photos of people wearing masks incorrectly.


#### Modeling

To effectively solve this problem, we took a direct approach. We built a three-tier classifier. We used a base model and trained a custom head layer that will separate faces into one of three classes: no mask, mask worn incorrectly, and mask worn correctly. We utilized a convolutional neural network that consisted of some trial and error. We settled on two layers at 16 nodes each, with 25 epochs, in addition to a pooling layer after each convolutional layer. We were faced with the limitations of computing, which required exploring Google Colab, and then heading back to the local environment after hitting RAM limitations. Our final model had 99% testing accuracy.


#### Video Detection

We utilized OpenCV 4.5.2, a library of programming functions mainly aimed at real-time computer vision, to access our user webcam. In addition, we used videostream from Imutils 0.5.4 which is a series of convenient functions to make basic image processing possible. These libraries simplified our ability to use our model to predict for the three classifications that we had focused on.

---

### Conclusions and Recommendations

Face Mask-Detection: Multi-Classification using Convolutional Neural Networks (CNN) model had a 99% accuracy on both the training and testing data. Measuring loss through categorical cross-entropy we were able to achieve successful results. Drilling into the individual classes, "With Mask" performed the best, with a .99 precision and .99 recall. This reinforced three main conclusions going forward. Our model's strength is predicting those who are masked and unmasked, but did not demonstrate the same precision with partially or improperly worn face masks.

Future areas of opportunity:

1). Expansive usage of more data to increase variety and diversity of material to train data on. In particular groups of people and additional face covering other than the typical blue surgical mask.

2). Simpler models are better due to computational resources and clarity. This can not be stressed enough in that it allows for replication and flexibility, which only improves the model.

3). Never start from zero! Past efforts of others cement our success. It is important to build on the shoulders of those that have produced successful results, because the collective effort is what can dramatically enhance the next iteration of this experiment.

---

### Additional Resources

**Internet Resources in relation to health and/or machine learning:**

[One Millisecond Face Alignment with an Ensemble of Regression Trees
Vahid Kazemi and Josephine Sullivan KTH, Royal Institute of Technology
Computer Vision and Active Perception Lab Teknikringen 14, Stockholm, Sweden](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf)

[Deep Learning Framework to Detect Face Masks
from Video Footage  Aniruddha Srinivas Joshi , Shreyas Srinivas Joshi Goutham Kanahasabai, Rudraksh Kapil, Savyasachi Gupta National Institute of Technology, Warangal, Telangana, Indi
https://arxiv.org/pdf/2011.02371v1.pdf](https://arxiv.org/pdf/2011.02371v1.pdf)

[Keras ImageDataGenerator and Data Augmentation by Adrian Rosebrock on July 8, 2019](https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/)
[Your Guide to Masks Updated Apr. 6, 2021](https://www.cdc.gov/coronavirus/2019-ncov/prevent-getting-sick/about-face-coverings.html)

**Blog posts:**

https://medium.com/@iselagradilla94/multi-task-cascaded-convolutional-networks-mtcnn-for-face-detection-and-facial-landmark-alignment-7c21e8007923

https://medium.com/analytics-vidhya/image-classification-with-mobilenet-cc6fbb2cd470

https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5

https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c
