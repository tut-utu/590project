from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, RepeatVector, Activation, Bidirectional
from keras.layers import concatenate
from keras.models import Model
from keras.optimizers import Adam, RMSprop


class keras_model():
    
    def __init__(self,max_len:int,embedding_size:int=256,input_size:int=4096,activation:str='relu'):
        self.sentence_len = max_len
        self._embedding_size = embedding_size
        self._image_model = Sequential()
        self._caption_model = Sequential()
        self._image_model.add(Dense(self._embedding_size, input_shape=(input_size,), activation=activation))
        self._image_model.add(RepeatVector(self.sentence_len))
        self._caption_model.add(Embedding(self.sentence_len,self._embedding_size, input_length=self.sentence_len))
        self._caption_model.add(LSTM(self._embedding_size, return_sequences=True))
        self._final_model = self._concat(self._image_model,self._caption_model)

    def _concat(self,model1,model2):
        concat_layer = concatenate([model1.output, model2.output],axis=1)
        x = Bidirectional(LSTM(self._embedding_size, return_sequences=False))(concat_layer)
        x = Dense(self.sentence_len)(x)
        out = Activation('softmax')(x)
        final_model = Model([model1.input, model2.input],[out])
        return final_model

    
    def compile(self,metrics,loss:str='categorical_crossentropy',optimizer='Adam'):
        if optimizer == 'Adam':
            opt = Adam()
        elif optimizer == 'RMSprop':
            opt = RMSprop()
        self._final_model.compile(loss=loss, optimizer=opt, metrics=metrics)

    def train(self,data_generator,epochs,callbacks):
        self._final_model.fit(data_generator,epochs=epochs,callbacks=callbacks)
    
    def summary(self):
        print(self._final_model.summary())

    def load_weights(self,filepath):
        self._final_model.load_weights(filepath=filepath)


