<<<<<<<
A neural chatbot using sequence to sequence model with
attentional decoder. This is a fully functional chatbot.

This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/


<h2>Usage</h2>

Step 1: create a data folder in your project directory, download
the Cornell Movie-Dialogs Corpus from 
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
Unzip it

Step 2: python data.py
<br>This will do all the pre-processing for the Cornell dataset.

Step 3:
python chatbot.py --mode [train/chat] <br>
If mode is train, then you train the chatbot. By default, the model will
restore the previously trained weights (if there is any) and continue
training up on that.

Step 4: The chat api
from model_chat.py import talk

talk("text") will return the generated text.
