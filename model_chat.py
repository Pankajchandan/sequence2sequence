import re
import os
from model import ChatBotModel
import tensorflow as tf
import data
import config
from chatbot import _check_restore_parameters, _find_right_bucket, run_step
import numpy as np


_, enc_vocab = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
inv_dec_vocab, _ = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))

model = ChatBotModel(True, batch_size=1)
model.build_graph()

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
_check_restore_parameters(sess, saver)
# Decode from standard input.
max_length = config.BUCKETS[-1][0]


# do sentence preprocessing
def sentence2id(vocab, line):
    return [vocab.get(token, vocab[b'<unk>']) for token in basic_tokenizer(line)]



def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile("\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words


# In[87]:


def _construct_response(output_logits, inv_dec_vocab):
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if config.EOS_ID in outputs:
        outputs = outputs[:outputs.index(config.EOS_ID)]
    # Print out sentence corresponding to outputs.
    response = " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])
    if len(outputs) == 0:
        return "i dont have a response as the first output generated was an EOS)"
    else:
        return response




def talk(text):
    global sess
    global max_length
    global saver
    global inv_dec_vocab
    global mmodel

    line = text
    if len(line) > 0 and line[-1] == '\n':
        line = line[:-1]
    if len(line) > 0 and line[-1] != '\n':
        line = line
    if line == '':
        return ("it's an empty string")
 
    # Get token-ids for the input sentence.
    token_ids = sentence2id(enc_vocab, str(line))
    if (len(token_ids) > max_length):
        return('Max length I can handle is:', max_length)
            
    # Which bucket does it belong to?
    bucket_id = _find_right_bucket(len(token_ids))
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, decoder_masks = data.get_batch([(token_ids, [])], bucket_id, batch_size=1)
    # Get output logits for the sentence.
    _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, True)
    
    response = _construct_response(output_logits, inv_dec_vocab)
    
    return response

