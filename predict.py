
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
max_seq_len = hparams['max_seq_len']

with open("comment.txt") as f:
    comment = f.read()

filter_lambda = lambda st: np.char.replace((' '.join(
                                     st.split()[0:max_seq_len]
                                                            )
                                                   ).lower(), '\\n', ' ')
comment = filter_lambda(comment)
print(comment)
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

sess = tf.Session()
bert_module =  hub.Module(bert_path, trainable = False)
tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    
vocab_file, do_lower_case = sess.run(
    [
        tokenization_info["vocab_file"],
        tokenization_info["do_lower_case"],
    ]
)
    
tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
tokens = ["[CLS]"]
tokens.extend(tokenizer.tokenize(re.sub('\s{2,}', ' ', re.sub('([^a-zA-Z])', r' \1 ', str(comment)))))
if len(tokens) > (max_seq_len - 1):
    tokens = tokens[:max_seq_len - 1]
tokens.append("[SEP]")
input_id = np.pad(np.array(tokenizer.convert_tokens_to_ids(tokens)), 
                     (0, max_seq_len-len(tokens)), "constant", constant_values=(0, 0))
segment_id = np.zeros(max_seq_len)
input_mask = np.zeros(max_seq_len)
input_mask[:len(tokens)] = 1

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

model = model_builder.build(True)

#======================================================================
#======================================================================
#======================================================================

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
sess.run(tf.local_variables_initializer())

predict_model = model_builder.model_predict
predict_pack = [[input_id], [input_mask], [segment_id]]
if hparams["feed_uppercase_info"]:
    predict_pack.append(input_uppercase_coeff)
print(ESCAPE_INFO + "Predicting" + ESCAPE_OFF)
result = predict_model.predict(predict_pack, verbose = 1)
result = np.array([x[0][1] for x in result])
np.save("comment_result", result)
