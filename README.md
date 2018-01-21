# Image-Recognition-Using-CNN

This project studies a convolutional neural network (CNN) architecture proposed by [Zeiler and Fergus](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf). The CNN is implemented on Keras from scratch. Following the described methodology, an image recognition is attempted on [VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) data set. To keep consistency with the reference paper, images with only unique and single labels are attemted for recognition.

<p align="center">
  <img src="https://github.com/tashrifbillah/Image-Recognition-Using-CNN/blob/master/CNN.JPG"/>
</p>

The obtained results for CNN classification is shown in our [project report](https://github.com/tashrifbillah/Object-Detection/blob/master/Tashrif_Billah_Object_Detection.pdf). However, the attached code may suffer some memory error. Also, multi-GPU implementation is incomplete. Please feel free to contribute and give a push.

# Code Execution Instruction

# Instruction for running code
Please see the following instruction for executing the project

1. Download the [VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

2. Run [extract_data.py](https://github.com/tashrifbillah/Object-Detection/blob/master/extract_data.py) that will extract
relevant information from the dataset i.e. image classes, indices, and bounding boxes and create a directory with 10 subimages (four corners + center, non-flipped and flipped). Please see [the paper](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf) for details about 10 subimages.

3. Follow the [tutorial](https://www.cs.columbia.edu/~smb/classes/f16/guide.pdf) to create your own virtual machine (VM) on Google Cloud GPU. Feel free to use any other GPU you might have access to.

4. Install necessary libraries on the VM.

6. Run [CNN_1.py](https://github.com/tashrifbillah/Image-Recognition-Using-CNN/blob/master/CNN_1.py) on the VM. This program will
train the CNN with weights for optimal image recognition. Feel free to play around with the train ratio. This might take 6-10 hours depending on the speed of your system. At the end, this program will print class wise accuracy of recognition by the trained CNN.

