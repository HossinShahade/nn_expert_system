from experta import Rule, L, NOT
from engine.facts import Problem, Architecture, Blueprint


class OutputLayerRules:

    # output types rules
    # PyTorch docs · pytorch.org
    @Rule(Problem(output_type='binary'), salience=10)
    def output_binary(self):
        self.declare(Blueprint(output_layer='Linear(last_hidden -> 1) no activation'))

    @Rule(Problem(output_type='multiclass'), salience=10)
    def output_multiclass(self):
        self.declare(Blueprint(output_layer='Linear(last_hidden -> num_classes) no activation'))

    @Rule(Problem(output_type='continuous_single'), salience=10)
    def output_continuous_single(self):
        self.declare(Blueprint(output_layer='Linear(last_hidden -> 1) no activation'))

    @Rule(Problem(output_type='continuous_multi'), salience=10)
    def output_continuous_multi(self):
        self.declare(Blueprint(output_layer='Linear(last_hidden -> number of outputs) no activation'))


    #=======================================================
    # cnn scratch \ cnn pretrained
    # Lin et al. 2013 · arxiv:1312.4400
    @Rule(Architecture(family=L('cnn_scratch') | L('cnn_pretrained')), salience=20)
    def cnn_pretrained_or_scratch_output(self):
        self.declare(Blueprint(output_layer='GlobalAvgPool2D -> Dropout -> Linear(channels -> num_classes)'))

    #------------------------------------
    # conv_autoencoder
    # Gong et al. 2019 · arxiv:1904.11294
    @Rule(Architecture(family='conv_autoencoder'), salience=20)
    def conv_autoencoder_output(self):
        self.declare(Blueprint(output_layer='ConvTranspose2D layers mirror encoder · Sigmoid on final layer if input in [0,1]'))

    #==============================
    # bert classify
    # Devlin et al. 2018 · arxiv:1810.04805
    @Rule(Architecture(family='bert'), Problem(task='classify'), salience=20)
    def bert_classify_output(self):
        self.declare(Blueprint(output_layer='[CLS] token (768d) -> Dropout(0.1) -> Linear(768 -> num_classes)'))

    # bert regression
    @Rule(Architecture(family='bert'), Problem(task='regress'), salience=20)
    def bert_regress_output(self):
        self.declare(Blueprint(output_layer='[CLS] token (768d) -> Dropout(0.1) -> Linear(768 -> 1)'))

    #-----------------------
    # gpt
    # Radford et al. 2019
    @Rule(Architecture(family='gpt'), salience=20)
    def gpt_output(self):
        self.declare(Blueprint(output_layer='Language model head already present · linear projection to vocab_size'))

    #==============================================
    # autoencoder
    # Bank et al. 2023 · arxiv:2003.05991
    @Rule(Architecture(family='autoencoder'), salience=20)
    def autoencoder_output(self):
        self.declare(Blueprint(output_layer='Final decoder layer -> Sigmoid if input in [0,1] · Tanh otherwise'))

    #----------------------------------------------
    # lstm autoencoder
    # Malhotra et al. 2016 · arxiv:1607.00148
    @Rule(Architecture(family='lstm_autoencoder'), salience=20)
    def lstm_autoencoder_output(self):
        self.declare(Blueprint(output_layer='decoder lstm reconstructs input sequence, linear projection back to input_dim'))

    #========================================
    # lstm classify
    # Graves et al. 2013 · arxiv:1303.5778
    @Rule(Architecture(family='lstm'), Problem(task='classify'), salience=20)
    def lstm_classify_output(self):
        self.declare(Blueprint(output_layer='last hidden state -> Linear(hidden_size -> num_classes)'))

    # lstm regression
    @Rule(Architecture(family='lstm'), Problem(task='regress'), salience=20)
    def regression_lstm_output(self):
        self.declare(Blueprint(output_layer='last hidden state -> Linear(hidden_size -> 1)'))

    #-----------------------
    # transformer_ts
    # Zhou et al. 2021 · arxiv:2012.07436
    @Rule(Architecture(family='transformer_ts'), salience=20)
    def transformer_ts_output(self):
        self.declare(Blueprint(output_layer='final hidden state -> Linear(model_dim -> forecast_horizon)'))

    #=========================
    # xgboost
    # Chen & Guestrin 2016 · arxiv:1603.02754
    @Rule(Architecture(family='xgboost'), salience=20)
    def xgboost_output(self):
        self.declare(Blueprint(output_layer='no output layer -- prediction handled internally by XGBoost API'))

    #------------------
    # logreg
    # Goodfellow et al. 2016
    @Rule(Architecture(family='logreg'), salience=20)
    def logreg_output(self):
        self.declare(Blueprint(output_layer='no output layer -- prediction handeled internally by sklearn API'))

    #------------
    # yolo
    # Redmon et al. 2016 · arxiv:1506.02640
    @Rule(Architecture(family='yolo'), salience=20)
    def yolo_output(self):
        self.declare(Blueprint(output_layer='detection head outputs [x,y,w,h,objectness,class_probs] per anchor'))

    #===================
    # wav2vec
    # Baevski et al. 2020 · arxiv:2006.11477
    @Rule(Architecture(family='wav2vec'), Problem(task='classify'), salience=20)
    def classify_wav2vec_output(self):
        self.declare(Blueprint(output_layer='pooled output -> Linear(hidden_size -> num_classes)'))

    #-------------
    # cnn_1d
    # LeCun et al. 1995
    @Rule(Architecture(family='cnn_1d'), Problem(task='classify'), salience=20)
    def cnn_1d_classify_output(self):
        self.declare(Blueprint(output_layer='GlobalAvgPool1D -> Linear(channels -> num_classes)'))

    @Rule(Architecture(family='cnn_1d'), Problem(task='regress'), salience=20)
    def regression_cnn_1d_output(self):
        self.declare(Blueprint(output_layer='GlobalAvgPool1D -> Linear(channels -> 1)'))

    @Rule(Architecture(family='gan'), salience=20)
    def gan_output(self):
        self.declare(Blueprint(output_layer='Generator head -> Tanh output, Discriminator head -> Sigmoid output for real/fake classification'))