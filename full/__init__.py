import tensorflow as tf
import numpy as np
import sys

class Model():
    def __init__(self,
                 hparams, 

                 target_weights, 
                 # array of size len(hparams["target_columns"]) with weights for balancing

                 identity_weights,
                 # array of size len(hparams["identity_columns"]) with weights for balancing

                 in_uppercase,
                 # Keras input layer with one value per comment             
 
                 identity_data,
                 # identity values
 
                 target_data,
                 # target values
                 
                 model_inputs,
                 # inputs layers of model, used for creating keras model

                 bert_output
                 # result of bert layer
                 # either (batch_size, max_seq_len, bert_dim) if sequential
                 # or     (batch_size, bert_dim)              if pooled
                 ):
        self.hparams = hparams
        print(identity_data.shape)
        if int(sys.argv[2]) == 0:
            self.target_weights = identity_weights
            self.names = hparams["identity_columns"]
            self.target = identity_data
        if int(sys.argv[2]) == 1:
            self.target_weights = target_weights
            self.names = hparams["target_columns"]
            self.target = target_data
        if int(sys.argv[2]) == 2:
            self.target_weights = np.concatenate((identity_weights, target_weights))
            self.names = np.concatenate((hparams["identity_columns"], hparams["target_columns"]))
            self.target = np.concatenate((identity_data, target_data), axis = 1)
        if int(sys.argv[2]) == 4:
            self.target_weights = [target_weights[0]]
            self.names = [hparams["target_columns"][0]]
            self.target = np.reshape(target_data[:, 0], (len(target_data), 1, 2))
        self.bert_output = bert_output
        self.model_inputs = model_inputs
    def build(self, predict=False):
        
        outputs_l = []
        for i in range(len(self.target_weights)):
            res = tf.keras.layers.Dense(512, activation='relu')(self.bert_output)
            res = tf.keras.layers.Dense(256, activation='relu')(res)
            res_single = tf.keras.layers.Dense(2, activation='softmax', name=self.names[i])(res)
            outputs_l.append(res_single)

        outputs = {}
        losses = {}
        metrics = {}
        for i in range(len(self.names)):
            losses[self.names[i]] = 'categorical_crossentropy'
            outputs[self.names[i]] = outputs_l[i]
            metrics[self.names[i]] = 'accuracy'
         
        model = tf.keras.models.Model(inputs = self.model_inputs, outputs = outputs)
        model.compile(optimizer="adam", 
                      loss=losses, 
                      metrics=metrics)
        ##############################
        ## post-trained predict stuff
        if predict:
            #model.load_weights(self.hparams["model_path"])
            self.model_predict = tf.keras.Model(inputs = self.model_inputs, 
                                                outputs = self.bert_output)
            print(self.model_predict.summary())
        else:
            print(model.summary())
        
        return model

    def labels(self, begin=0, end=-1):
        labels = {}
        for i in range(len(self.names)):
            labels[self.names[i]] = self.target[begin:end, i]
        return labels
    def class_weights(self):
        weights = {}
        for i in range(len(self.names)):
            weights[self.names[i]] = {0:self.target_weights[i], 1:1.0}
        return weights

