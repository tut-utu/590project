Here we explain how the framework can be used to predict sentences for arbitrary images.

1. Copy all images you want to predict for to one folder. For example, this folder contains multiple, collected from Reddit's r/photoshopbattles.
2. Extract the CNN features for all images with the Matlab script provided in `matlab_features_reference`, and as described with the Readme file in that folder. I want to eventually allow people to extract features with Python but for now it is needed to go through Matlab. The Matlab script needs to be pointed to a file `tasks.txt` that you should create in the same folder. I show the example in this folder as well: It lists the images that you wish to process in some order. The Matlab file will extract features into a file called `vgg_feats.mat`.
3. Now that we have the features we can run the prediction! Use the script `predict_on_images.py`. The script takes the path to a model checkpoint and the path to the folder that holds the images, the `tasks.txt`, and the features `vgg_feats.mat`. Example invocation is `python predict_on_images.py lstm_model.p -r example_images`. The script will write the html file `result.html` which you can use to visualize the results in your browser.

Note that the models are trained on a particular dataset (e.g. COCO dataset), so if you show them images they haven't seen during their training time then they will produce garbage. Along with the sentence predictions I'm also showing the log probabilities. When this is low (e.g. -10), this means that the model is confused about the image and likely won't make very good predictions. Conversely, higher numbers (such as -7) indicate that the model is relatively more confident in the outcome.