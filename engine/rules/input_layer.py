from experta import Rule, NOT, OR, L, salience
from engine.facts import Problem, Architecture, Blueprint



#____________________________________
#mlp
#Goodfellow et al. 2016 · deeplearningbook.org


@Rule(Architecture(family='mlp'),Problem(input_shape='small'),salience=10)
def mlp_small_input(self):
    self.declare(Blueprint(input_layer='linear(features->64)'))
@Rule(Architecture(family='mlp'),Problem(input_shape='medium'),salience=10)
def mlp_medium_input(self):
    self.declare(Blueprint(input_layer='linear(features->256)'))
@Rule(Architecture(family='mlp'),Problem(input_shape='large'),salience=10)
def mlp_input_large(self):
    self.declare(Blueprint(input_layer='linear(fetures->512)',notes_input='consider PCA to reduce diminsionality first'))
    
#______________________________________
#cnn scratch
#He et al. 2015 · arxiv:1512.03385

@Rule(Architecture(family='cnn_scratch'),Problem(input_shape='small'),salience=10)
def cnn_scratch_input_small(self):
    self.declare(Blueprint(input_shape='conv2D(16,3X3,padding=1)->BN-> ReLU, no early pooling'))
@Rule(Architecture(family='cnn_scratch'),Problem(input_shape='medium'),salience=10)
def cnn_scratch_medium_input(self):
    self.declare(Blueprint(input_layer='conv2D(32,3X3,padding=1)->BN->ReLU->MAXPool(2X2)'))
@Rule(Architecture(family='cnn_scratch'),Problem(input_shape='large'),salience=10)
def cnn_sscratch_large_input(self):
    self.declare(Blueprint(input_layer='conv2D(64,7X7,stride=2)->BN->RelU->MAXPool(3X3,stride=2)'))
#____________________________________________________
#cnn pretained
#Kornblith et al. 2019 · arxiv:1805.08974
@Rule(Architecture(family='cnn_pretrained'),salience=10)
def cnn_pretrained_input(self):
    self.declare(Blueprint(input_layer='resize to 224X224 , normalize with imagenet mean =[0.229,0.224,0.225]'))
#--------------------------------------------
#conv_autoencoder
#Gong et al. 2019 · arxiv:1904.11294
@Rule(Architecture(family='conv_autoencoder'),Problem(input_shape='small'),salience=10)
def conv_autoencoder_input_small(self):
    self.declare(Blueprint(input_layer='conv2D(16,3X3,padding=1)->RelU'))
@Rule(Architecture(family='conv_autoencoder'),Problem(input_shape='medium'),salience=10)
def conv_autoencoder_input_medium(self):
    self.declare(Blueprint(input_layer='conv2D(32,3X3,stride=2)->RelU'))
@Rule(Architecture(family='conv_autoencoder'),Problem(input_shape='large'),salience=10)
def conv_autoencoder_input_large(self):
    self.declare(Blueprint(input_layer='conv2D(64,3X3,stride=2)->RelU'))
#----------------------------------------------------------
#lstm
#Hochreiter & Schmidhuber 1997
@Rule(Architecture(family='lstm'),Problem(modality='timeseries'),salience=10)
def lstm_input_timeseries(self):
    self.declare(Blueprint(input_layer='RAW features pre timestep no embedding needed'))
@Rule(Architecture(family='lstm'),Problem(modality='text'),salience=10)
def lstm_input_text(self): #Mikolov et al. 2013 · arxiv:1301.3781
    self.declare(Blueprint(input_layer='Embedding(vocab_size->128)->lstm'))
#---------------
#lstm_autoencoder
#Malhotra et al. 2016 · arxiv:1607.00148 
       
@Rule(Architecture(family='lstm_autoencoder'),salience=10)
def lstm_autoencoder_input(self):
    self.declare(Blueprint(input_layer='RAW sequence features per time step no embedding'))
#----------------------------
#bert
#Devlin et al. 2018 · arxiv:1810.04805
@Rule(Architecture(family='bert'),salience=10)
def input_bert(self):
    self.declare(Blueprint(input_layer='Tokenizer->token_ids+ attention mask, max length=512'))
#-----------------
#gpt
#Radford et al. 2019
@Rule(Architecture(family='gpt'),salience=10)
def gpt_input(self):
    self.declare(Blueprint(input_layer='Tokenizer=> token_id, max leangth varies by model variant'))
#----------------
#transformer ts
#Zhou et al. 2021 · arxiv:2012.07436

@Rule(Architecture(family='transformer_ts'),salience=10)
def transformer_ts_input(self):
    self.declare(Blueprint(input_layer='linear projection of input features to mpodel_dim.add positional encoding'))


#____________________________
#autoencoder
#Bank et al. 2023 · arxiv:2003.05991

#small
@Rule(Architecture(family='autoencoder'),Problem(input_shape='small'),salience=10)
def auto_encoder_small_input(self):
    self.declare(Blueprint(input_layer='linear(features->64)'))
#medium
@Rule(Architecture(family='autoencoder'),Problem(input_shape='medium'),salience=10)
def autoencoder_medium_input(self):
    self.declare(Blueprint(input_layer='linear(fetures->256)'))
#large
@Rule(Architecture(family='autoencoder'),Problem(input_shape='large'),salience=10)
def autoencoder_large_input(self):
    self.declare(Blueprint(input_layer='linear(features->512)'))
#_________________________________--#
#xgboost
#Chen & Guestrin 2016 · arxiv:1603.02754
@Rule(Architecture(family='xgboost'),salience=10)
def xgboost_input(self):
    self.declare(Blueprint(input_layer='RAw feature victor, no layer needed ',notes_input='apply standerd scaler befor fitting'))

#_____________________________
#logreg
#Goodfellow et al. 2016
@Rule(Architecture(family='logreg'),salience=10)
def logreg_input(self):
    self.declare(Blueprint(input_layer='RAw feature vector',notes_input='apply standerdscaler   logistic regresssion is sensitive to scale'))

#_________
#yolo
#Redmon et al. 2016 · arxiv:1506.02640
@Rule(Architecture(family='yolo'),salience=10)
def yolo_input(self):
    self.declare(Blueprint(input_layer='resize to 640X640 noermalize pixel value to [0,1]'))
#_____________
#wav2vec
#Baevski et al. 2020 · arxiv:2006.11477
@Rule(Architecture(family='wav2vec'),salience=10)
def wav2vec_input(self):
    self.declare(Blueprint(input_layer='raw waveform at 16KHz no preprocessing needed'))
#-----------------------------
#cnn 1D
#LeCun et al. 1995
@Rule(Architecture(family='cnn_1d'),salience=10)
def cnn_1d_input(self):
    self.declare(Blueprint(input_layer='raw waveform or MFCC features n_mfcc=40'))

