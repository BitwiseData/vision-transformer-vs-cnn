# Deep-Learning

# Vision Transformers vs. Convolutional Neural Networks

# **Overview, Task, and Data Description**

Natural language processing has come a long way in the previous five years thanks in large part to transformers. The use of transformers in computer vision is a relatively new development. Whether vision transformers or convolutional neural networks are superior for classification is currently being investigated. In light of this, your assignment is to develop and evaluate three distinct transformer and CNN sizes.

There are three accessible datasets that may be used to compare and contrast vision transformers and CNNS. All of these data sets may be downloaded for free from the TensorFlow repository. To access the appropriate Tensorflow Datasets, click on the URLs provided below.

## **Programming Requirements**

For this project, you will need to implement three different sizes of vision transformers and 
CNNs. How you define size is up to you. This can include, but is not limited to, number of 
layers, number of neurons, or filters. The names of the size are tiny, small, and base with base 
being larger than small, and small being larger than tiny. You also need to decide all hyperparameters, including but not limited to, loss, optimizer, learning rates, epochs, etc. The 
following points are requirements for the project.

* Must be written in python using Tensorflow/Keras. You will need to create two scripts for 
this project
a. train.py – train vision transformer and CNN on dataset
b. test.py – test vision transformer and CNN on dataset
You can have other scripts, but you must have at least these two. 

* For this project you will be required to use the functional API, for at least the transformer. This is due to the creation of residual connections in the vision transformers (see functional API link above for examples on how to include residual connections in Keras). Note that you can still use the sequential API for your CNNs.

* The following metrics, to evaluate the results of your test data, must be print to the 
command line.
a. Accuracy
b. Recall (macro and micro)
c. Precision (macro and micro)
d. F1 score (macro and micro) 

* Your python scripts must take two command line parameters.
a. Size of models to build/load (tiny, small, base)
b. Dataset to train/evaluate (cifar10, cifar100, fashion_mnist)

* Your train.py file must save both the vision transformer and CNN models to .h5 format. 
Save them to the same directory as your scripts are running from. They must be saved 
with specific file names based on the model size, as shown below.
a. Tiny – tiny_vision_transformer.h5; tiny_CNN.h5
b. Small – small_vision_transformer.h5; small_CNN.h5
c. Base – base_vision_transformer.h5 ; base_CNN.h5
Note, that for your experiments, you need to evaluate cifar10, cifar100, and 
fashion_mnist. This would result in 18 different models. You need to submit 6 models 
with your final project submission. These specific models must be trained on cifar100. Do 
not submit models that have been trained on the other datasets.
Example for training is given below.
python train.py tiny cifar100
This would train your tiny version of a vision transformer and CNN on cifar100. It would 
save two models: tiny_vision_transformer.h5 and tiny_CNN.h5.

* Your test.py file must load both the vision transformer and CNN model based on the 
model size. Example for testing is given below.
python test.py tiny cifar100
This would evaluate your tiny models (vision transformer and CNN) on cifar100. It will 
load tiny_vision_transformer.h5 and tiny_CNN.h5. The evaluation metrics for both the 
vision transformer and CNN need to be print to the screen. Be clear on which metrics are 
for which model.

### **Paper Requirements**


For the paper, you will need to turn in a 4-6 page plus references (unlimited pages for 
references) paper in the IEEE double column format. The 6 pages is a hard limit. You can NOT 
have any content, other than references, on the 7
th page or more. You can find Latex and 
Word templates on Canvas. Do not change the templates (e.g., font size, etc.) This paper 
needs to be written like a scientific article that would be submitted to a conference so they 
all should have the same format. The same sections/ideas will apply in this paper, and you 
need to at least have the following sections (you can add more if needed). Also note, more 
details than what are given here should be included. This is just a brief overview of what to 
include at a minimum.

1. Abstract (~150 words)

2. Introduction 

3. Related works (related works can also be combined with intro)
a. Give a brief overview of at least 10 papers that are related to your work. At least 
five of the papers need to be on vision transformers. The other five can be on 
CNNs. As a note, at least two papers need to specifically mention CNNs as the core 
architecture.

4. Method
a. Describe your tiny, small, and base vision transformer and CNN architectures.
b. How did you decide what tiny, small, and base are? Is this based on the number 
of layers, neurons, or something else?

5. Experiments and results
a. What datasets did you use? Provide a brief description of each of them including 
number of training/testing samples.
b. What experiments did you run? This would include talking about comparing vision
transformers and the results. Talk about each of the models. For each of the sizes 
and datasets which are better, vision transformers or CNNs?
c. Your experiments and results section must have at least the following figures or 
tables. You decide how to do this. You must show the following comparisons for 
all sizes (tiny, small, base) on all datasets.
i. Comparing loss of transformers and CNNs, for each size, on each dataset. 
ii. Comparing accuracy of transformers and CNNs, for each size, on each 
dataset.

6. Conclusion

7. References
a. At least 10 references needed. Note all references must be cited in the paper. Do 
not include a reference that does not have a citation in the paper.
