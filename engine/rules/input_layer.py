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