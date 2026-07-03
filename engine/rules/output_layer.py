from experta import *
from engine.facts import Problem, Architecture, Blueprint


#output types rules
#PyTorch docs · pytorch.org
@Rule(Problem(output_type='binary'),salience=10)
def output_binary(self):
    self.declare(Blueprint(output_layer='linear(last hiddden->1no activation'))
@Rule(Problem(output_type='multiclass'),salience=10)
def output_multiclass(self):
    self.declare(Blueprint(output_layer='linear(last hidden->num_classes)no activation'))
@Rule(Problem(output_type='continuous_single'),salience=10)
def output_continuous_single(self):
    self.declare(Blueprint(output_layer='linear(last hidden->1no activation'))
@Rule(Problem(output_type='continuous_multi'),salience=10)
def output_continuous_multi(self):
    self.declare(Blueprint(output_layer='linear(last hidden->number of outputs)no activation'))


#=======================================================
#cnn scratch\cnn pretrained
#Lin et al. 2013 · arxiv:1312.4400
@Rule(Architecture(Family=L('cnn_scratch')|L('cnn_pretrained')),salience=15)
def cnn_pretrained_or_scratch_output(self):
    self.declare(Blueprint(output_layer='global avg pool 2D -> dropout->linear(channels-> num of classes)'))
#===================================================
#conv_autoencoder
#Gong et al. 2019 · arxiv:1904.112947
@Rule(Architecture(family='conv_autoencoder'),salience=10)
def conv_autoencoder_output(self):
    self.declare(Blueprint(output_layer='ConvTranspose2D layers mirror encoder · Sigmoid on final layer if input in [0,1]'))
#==============================
#bert
#clssify
#Devlin et al. 2019 · arxiv:1810.04805
@Rule(Architecture(family='bert'),Problem(task='classify'),salience=10)
def bert_classify_output(self):
    self.declare(Blueprint(output_layer='[CLS] token (768d) → Dropout(0.1) → Linear(768 → num_classes)'))
#regression
@Rule(Architecture(family='bert'),Problem(task='regress'),salience=10)
def bert_regress_output(self):
    self.declare(Blueprint(output_layer='[CLS] token (768d) → Dropout(0.1) → Linear(768 → 1)'))
#=======================
#gpt
#Radford et al. 2019
@Rule(Architecture(family='gpt'),salience=10)
def gpt_output(self):
    self.declare(Blueprint(output_layer='Language model head already present · linear projection to vocab_size'))
#==============================================
#autoencoder
#Bank et al. 2023 · arxiv:2003.05991
@Rule(Architecture(family='autoencoder'),salience=10)
def autoencoder_output(self):
    self.declare(Blueprint(output_layer='Final decoder layer → Sigmoid if input in [0,1] · Tanh otherwise'))
#==============================================================
#lstm auto encoder
#Malhotra et al. 2016 · arxiv:1607.00148
@Rule(Architecture(family='lstm_autoencoder'),salience=10)
def lstm_autoencoder_output(self):
    self.declare(Blueprint(outout_layer='decoddr lstm reconstructs input sequencr, linear projection back to input dim'))
#========================================
#lstm
#Graves et al. 2013 · arxiv:1303.5778
#classify
@Rule(Architecture(family='lstm'),Problem(task='classify'),salience=10)
def lstm_classify_output(self):
    self.declare(Blueprint(output_layer='last hiddens state-> linear(hidden size->num of classes)'))
#regression
@Rule(Architecture(family='lstm'),Problem(task='regress'),salience=10)
def regression_lstm_output(self):
    self.declare(Blueprint(output_layer='last hidden state-> linear(hidden size=1)'))
#=======================
#transformer_ts
#Zhou et al. 2021 · arxiv:2012.07436
@Rule(Architecture(family='transformer_ts'),salience=10)
def transformer_ts_output(self):
    self.declare(Blueprint(output_layer='final hidden state-> linear(model dim-> forcast_horizon)'))
#=========================
#xgboost
#Chen & Guestrin 2016 · arxiv:1603.02754
@Rule(Architecture(family='xgboost'),salience=10)
def xgboost_output(self):
    self.declare(Blueprint(output_layer='no outputlayer--- prediction handeed internally by XGBoost API'))
#==================
#logreg
#Goodfellow et al. 2016
@Rule(Architecture(family='logreg'),salience=10)
def logreg_output(self):
    self.declare(Blueprint(output_layer='no output layer--- predition handeled internally by SKlearn API'))
#============
#yolo
#Redmon et al. 2016 · arxiv:1506.02640
@Rule(Architecture(family='yolo'),salience=10)
def yolo_output(self):
    self.declare(Blueprint(output_layer='detection head outputs[x,y,w,h,objectness,class probs]per anchor'))
#===================
#wav2vec
#Baevski et al. 2020 · arxiv:2006.11477
@Rule(Architecture(family='wav2vec'),Problem(task='classify'),salience=10)
def classify_wav2vec_output(self):
    self.declare(Blueprint(output_layer='pooled output-> linear(hidden_size->num of classis)'))
#=============
#cnn 1d
#LeCun et al. 1995
@Rule(Architecture(family='cnn_1d'),Problem(task='classify'),salience=10)
def cnn_1d_classify_output(self):
    self.declare(Blueprint(output_layer='global avg pool 1d -> linear(channels-> num of classes)'))
@Rule(Architecture(family='cnn_1d'),Problem(task='regress'),salience=10)
def regression_cnn_1d_output(self):
    self.declare(Blueprint(output_layer='global avg pool 1d -> linear (channels->1)'))
    
    
    
    