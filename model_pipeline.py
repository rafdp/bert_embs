
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import numpy as np
from bert.tokenization import FullTokenizer
from tensorflow.keras import backend as K
import time
import re
from keras import metrics
import sys
import json
import os
hparams = json.load(open(sys.argv[1]))
ESCAPE_INFO = '\033[38;5;209m'
ESCAPE_TITLE = '\033[38;5;123m'
ESCAPE_DATA = '\033[38;5;72m'
ESCAPE_FILE = '\033[38;5;118m'
ESCAPE_OFF = '\033[0m'

MAX_SEQ_LEN = hparams['max_seq_len']
BATCH_SIZE = hparams['batch_size']
BERT_DIM = 768

Model = __import__(hparams['name']).Model

sess = tf.Session()
K.set_session(sess)

slice_size = -1

preprocessed_data = np.load(hparams["data_path"] + '.npy', 
                            allow_pickle=True)[0]
target_weights = preprocessed_data["target_weights"]
identity_weights = preprocessed_data["identity_weights"]
input_uppercase_coeff = preprocessed_data["uppercase_coeff"]
identity_data = np.array([[[1, 0] if val < 0.5 else [0, 1] for val in col] for col in preprocessed_data["identity_data"]], dtype=np.int32)
target_data = np.array([[[1, 0] if val < 0.5 else [0, 1] for val in col] for col in preprocessed_data["target_data"]], dtype=np.int32)
input_ids   = preprocessed_data["bert_input_data"]["input_ids"]
segment_ids = preprocessed_data["bert_input_data"]["segment_ids"]
input_masks = preprocessed_data["bert_input_data"]["input_masks"]

"""
input_ids = input_ids[:slice_size]
segment_ids = segment_ids[:slice_size]
input_masks = input_masks[:slice_size]
"""

class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        **kwargs,
    ):
        self.trainable = True
        self.bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        ) 

        
        trainable_vars = self.bert.variables
        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
        #trainable_vars = [var for var in trainable_vars if """("output/dense/" in var.name and "attention" not in var.name) or""" "pooler" in var.name] 
        trainable_vars = [var for var in trainable_vars if "pooler" in var.name] 
        # Select how many layers to fine tune
        for v in trainable_vars:
            print(ESCAPE_FILE + v.name)
        print(ESCAPE_OFF)
        trainable_vars = trainable_vars[-30 :]
        for var in trainable_vars:
            self._trainable_weights.append(var)
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
                          )
        if hparams["bert_output"] == "sequential":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["sequence_output"]
        elif hparams["bert_output"] == "pooled":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["pooled_output"]
        else:
            raise ValueError("Invalid bert_output (%s) in hparams" % hparams["bert_output"])
        return result

    def compute_output_shape(self, input_shape):
        if hparams["bert_output"] == "sequential":
            return (input_shape[0], MAX_SEQ_LEN, BERT_DIM)
        elif hparams["bert_output"] == "pooled":
            return (input_shape[0], BERT_DIM)
        else:
            raise ValueError("Invalid bert_output (%s) in hparams" % hparams["bert_output"])

print("Building model")

K.set_session(sess)

in_id = tf.keras.layers.Input(batch_size = BATCH_SIZE, 
                              shape=(MAX_SEQ_LEN,), 
                              name="input_ids")
in_mask = tf.keras.layers.Input(batch_size = BATCH_SIZE, 
                                shape=(MAX_SEQ_LEN,), 
                                name="input_masks")
in_segment = tf.keras.layers.Input(batch_size = BATCH_SIZE, 
                                   shape=(MAX_SEQ_LEN,), 
                                   name="segment_ids")
if hparams["feed_uppercase_info"]:
    in_uppercase = tf.keras.layers.Input(batch_size = BATCH_SIZE, 
                                         shape=(1,),
                                         name="uppercase_coeff")
else:
    in_uppercase = None
bert_inputs = [in_id, in_mask, in_segment]

bert_output = BertLayer()(bert_inputs)
#exit()
if hparams["feed_uppercase_info"]:
    model_inputs = [in_id, in_mask, in_segment, in_uppercase]
else:
    model_inputs = bert_inputs

model_builder = Model(hparams,
                      target_weights, 
                      identity_weights,
                      in_uppercase,
                      identity_data,
                      target_data,
                      model_inputs,
                      bert_output)

if "predict" not in hparams:
    hparams["predict"] = False

model = model_builder.build(hparams["predict"])
#model = model_builder.build(True)

"""
out = tf.keras.layers.Conv1D(128, 2, activation='relu', padding='same')(bert_output)
out = tf.keras.layers.MaxPooling1D(5, padding='same')(out)
out = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(out)
out = tf.keras.layers.MaxPooling1D(5, padding='same')(out)
out = tf.keras.layers.Conv1D(128, 4, activation='relu', padding='same')(out)
out = tf.keras.layers.MaxPooling1D(40, padding='same')(out)
out = tf.keras.layers.Flatten()(out)
print(out.shape)
out = tf.keras.layers.Dropout(0.3)(out)
dense_res = tf.keras.layers.Dense(128, activation='relu')(out)
binary_output = tf.keras.layers.Dense(2, activation='softmax')(dense_res)

binary_output = tf.keras.layers.Dense(2, activation='softmax')(dense_res)
opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model = tf.keras.models.Model(inputs = bert_inputs, outputs = binary_output)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy", metrics.categorical_accuracy]) 
"""
#======================================================================
#======================================================================
#======================================================================

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
sess.run(tf.local_variables_initializer())

if hparams["predict"]:
    predict_model = model_builder.model_predict
    predict_pack = [input_ids, input_masks, segment_ids]
    if hparams["feed_uppercase_info"]:
        predict_pack.append(input_uppercase_coeff)
    print(ESCAPE_INFO + "Predicting" + ESCAPE_OFF)
    embeddings = \
        predict_model.predict(predict_pack, verbose = 1)
    print(ESCAPE_INFO + "Saving" + ESCAPE_OFF)
    np.save(hparams['data_path'] + '/embs', embeddings)
    print(ESCAPE_INFO + "Predict done" + ESCAPE_OFF)
if not hparams["predict"]:
    train_len = int(len(input_ids)*hparams['train_test_coeff'])
    train_len = (train_len//BATCH_SIZE)*BATCH_SIZE
    test_len = len(input_ids)-train_len
    test_len = (test_len//BATCH_SIZE)*BATCH_SIZE
    if len(sys.argv) > 3:
        train_len = BATCH_SIZE
        test_len = BATCH_SIZE
    
    train_input_ids = input_ids[:train_len]
    train_input_masks = input_masks[:train_len]
    train_segment_ids = segment_ids[:train_len]
    test_input_ids = input_ids[train_len:train_len + test_len]
    test_input_masks = input_masks[train_len:train_len+test_len]
    test_segment_ids = segment_ids[train_len:train_len+test_len]
    train_pack = [train_input_ids, 
                  train_input_masks, 
                  train_segment_ids] 
    test_pack = [test_input_ids, 
                 test_input_masks, 
                 test_segment_ids]
    if hparams["feed_uppercase_info"]:
        train_input_uppercase_coeff = input_uppercase_coeff[:train_len]
        test_input_uppercase_coeff  = input_uppercase_coeff[train_len:train_len+test_len]
        train_pack.append(train_input_uppercase_coeff)
        test_pack.append(test_input_uppercase_coeff)
         
    results = []
    
    labels_train = model_builder.labels(0, train_len)
    labels_test = model_builder.labels(train_len, train_len + test_len)
    if hparams['loss_weighting']:
        class_weight = model_builder.class_weights() 
    else:
        class_weight = None
    
    if os.path.exists(hparams["model_path"] + sys.argv[2]):
        print(ESCAPE_DATA + "Loading model weights")
        model.load_weights(hparams["model_path"] + sys.argv[2])
    if not hparams["predict"]:
        print(ESCAPE_INFO + ("Fitting model %s" % hparams['name']) + ESCAPE_OFF)
        callb = tf.keras.callbacks.TensorBoard(log_dir=hparams['data_path']+'/Graph', histogram_freq=0,  
                  write_graph=True, write_images=True, update_freq=6400)
        model.fit(train_pack, 
                  labels_train,
                  validation_data=(test_pack, 
                                   labels_test),
                      epochs=1,
                      batch_size=BATCH_SIZE,
                      class_weight=class_weight,
                  callbacks=[callb])
    
        model.save_weights(hparams["model_path"] + sys.argv[2])
        
    print(ESCAPE_INFO + ("Fit for model %s ended" % hparams['name']) + ESCAPE_OFF)
    predict_model = model_builder.model_predict
    
    predict_pack = [input_ids, input_masks, segment_ids]
    if hparams["feed_uppercase_info"]:
        predict_pack.append(input_uppercase_coeff)
    print(ESCAPE_INFO + "Predicting" + ESCAPE_OFF)
    embeddings = \
        predict_model.predict(predict_pack, verbose = 1)
    print(ESCAPE_INFO + "Saving" + ESCAPE_OFF)
    np.save(hparams['data_path'] + '/embs' + sys.argv[2], embeddings)
    print(ESCAPE_INFO + "Predict done" + ESCAPE_OFF)
    
    
