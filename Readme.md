This project contains *Python+numpy* source code for learning **Multimodal Recurrent Neural Networks** that describe images with sentences. The project convert karpathy's [neuraltalk](https://github.com/karpathy/neuraltalk) from python 2 to python 3 version, and add TensorFlow Keras model.

## Overview
The pipeline for the project looks as follows:

- The **input** is a dataset of images and 5 sentence descriptions that were collected with Amazon Mechanical Turk. In particular, this code base is set up for [Flickr8K](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [Flickr30K](http://shannon.cs.illinois.edu/DenotationGraph/), and [MSCOCO](http://mscoco.org/) datasets. 
- In the **training stage**, the images are fed as input to LSTM and the LSTM is asked to predict the words of the sentence, conditioned on the current word and previous context as mediated by the hidden layers of the neural network.
- In the **prediction stage**, a witheld set of images is passed to RNN and the RNN generates the sentence one word at a time. The results are evaluated with **BLEU score**. The code also includes utilities for visualizing the results in HTML.

## Dependencies
**Python 3**, modern version of **numpy/scipy**, **perl** (if you want to do BLEU score evaluation), **argparse** module. Most of these are okay to install with **pip**. To install all dependencies at once, run the command `pip install -r requirements.txt`

## Getting started

1. **Get the code.** `$ git clone` the repo and install the Python dependencies
2. **Get the data.** Download the `data/` folder from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/). Also, this download does not include the raw image files, so if you want to visualize the annotations on raw images, you have to obtain the images from Flickr8K / Flickr30K / COCO directly and dump them into the appropriate data folder.
3. **Train the model.** Run the training `$ python run.py` (see many additional argument settings inside the file) and wait. Checkpoints are written into `saved_model/` periodically.
4. **Evaluate model checkpoints.** To evaluate a checkpoint from `saved_model/`, run the `eval_sentence_predctions.py` script and pass it the path to a checkpoint.
5. **Visualize the predictions.** Use the included html file `visualize_result_struct.html` to visualize the JSON struct produced by the evaluation code. This will visualize the images and their predictions. Note that you'll have to download the raw images from the individual dataset pages and place them into the corresponding `data/` folder. NOTE: You have to run a local webserver (e.g. $ python -m SimpleHTTPServer 8123) and then open `visualize_result_struct.html` in order for the .html file to visualize the result properly.
