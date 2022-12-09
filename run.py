
from keras_version.model import keras_model
from keras_version.data_generator import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import argparse
import json
from keras.callbacks import ModelCheckpoint

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

def main(params):
    fix_gpu()
    #data generator parameters
    dataset = params['dataset']
    word_count_threshold=params['word_count_threshold']
    batch_size=params['batch_size']
    embedding_size=params['embedding_size']
    input_size=params['input_size']
    activation=params['activation']



    #model parameters
    metrics = params['metrics']
    loss=params['loss']
    optimizer=params['optimizer']
    
    epochs=params['epochs']
    mode = params['mode']
    checkpoint_filepath = params['checkpoint_filepath']
    model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    save_best_only=False,
    save_freq=1000)


    data = data_generator(dataset=dataset)
    data.build_vocab(word_count_threshold=word_count_threshold)
    generatort = build_generator(data._wordtoix,data._max_len,batch_size,dataset)
    model = keras_model(data._max_len,data._vocab_len,embedding_size=embedding_size,input_size=input_size,activation=activation)
    model.compile(metrics=metrics,loss=loss,optimizer=optimizer)
    if mode == 'exist':
        model.load_weights(checkpoint_filepath)
    model.train(generatort,epochs=epochs,callbacks=[model_checkpoint_callback])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #data generator parameters
    parser.add_argument('-d', '--dataset', dest='dataset', type=str, default='flickr8k', help='Name of the dataset, e.g. flicker8k, flicker30k, COCO')
    parser.add_argument('-t', dest='word_count_threshold', type=int, default=5, help='Threshold that decides if to discard a word based on its number of appearance')
    parser.add_argument('-b','--batch_size', dest='batch_size', type=int, default=256, help='Size of each batch')

    #model parameters
    parser.add_argument('-e','--embedding_size',dest='embedding_size',type=int,default=258, help='Size of embedding space')
    parser.add_argument('-i','--input_size',dest='input_size',type=int,default=4096, help='Size of the image feature')
    parser.add_argument('-a','--activation',dest='activation',type=str,default='relu', help='Activation function')
    parser.add_argument('-m','--metrics',dest='metrics',nargs='+',type=str,default=['Accuracy'], help='Model evaluation metrics')
    parser.add_argument('-l','--loss',dest='loss',type=str,default='categorical_crossentropy', help='Size of the image feature')
    parser.add_argument('-o','--optimizer',dest='optimizer',type=str,default='Adam', help='optimizer')
    parser.add_argument('-s','--epochs',dest='epochs',type=int,default='1', help='number of epochs')
    parser.add_argument('-f','--filepath',dest='checkpoint_filepath',type=str,default=r'./saved_model/checkpoint', help='checkpoint_filepath')
    parser.add_argument('--mode',dest='mode',type=str,default='new', help='Train new model or on exist model. Default: new mode')



    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed parameters:')
    print(json.dumps(params, indent = 2)) 
    main(params)