
from keras_version.model import keras_model
from keras_version.data_generator import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


fix_gpu()

data = data_generator()
data.build_vocab()
generatort = build_generator(data._wordtoix,data._max_len)
model = keras_model(data._max_len)
model.compile(metrics=['Accuracy'],loss='categorical_crossentropy',optimizer='Adam')
model.train(generatort,epochs=1)