from experta import Rule, L, NOT
from engine.facts import Problem, Architecture, Blueprint


class DropoutRules:

    # dropout rate rules
    # based on dataset size, these fire for every family
    # source: Srivastava et al. 2014 · JMLR 15

    @Rule(Problem(dataset_size=L('tiny') | L('small')))
    def dr1_rate_small(self):
        self.declare(Blueprint(dropout_rate='0.4 to 0.5'))

    @Rule(Problem(dataset_size='medium'))
    def dr2_rate_medium(self):
        self.declare(Blueprint(dropout_rate='0.3'))

    @Rule(Problem(dataset_size='large'))
    def dr3_rate_large(self):
        self.declare(Blueprint(dropout_rate='0.1 to 0.2'))

    #==============================================
    # dropout placement rules
    # each family hits exactly one rule

    # mlp
    # Source: Srivastava et al. 2014 · JMLR 15
    @Rule(Architecture(family='mlp'))
    def dr4_mlp_placement(self):
        self.declare(Blueprint(dropout_placement='after every hidden layer'))

    # cnn scratch
    @Rule(Architecture(family='cnn_scratch'))
    def dr5_cnn_scratch_placement(self):
        self.declare(Blueprint(dropout_placement='before final linear layer only'))

    #==============================================
    # cnn pretrained
    # Source: Yosinski et al. 2014 · arxiv:1411.1792
    @Rule(Architecture(family='cnn_pretrained'))
    def dr6_cnn_pretrained_placement(self):
        self.declare(Blueprint(dropout_placement='before classification head only, rate fixed at 0.3'))

    # conv_autoencoder
    # Source: Gong et al. 2019 · arxiv:1904.11294
    @Rule(Architecture(family='conv_autoencoder'))
    def dr7_conv_autoencoder_placement(self):
        self.declare(Blueprint(dropout_placement='between encoder blocks only'))

    #==============================================
    # lstm / lstm_autoencoder
    # Source: Zaremba et al. 2014 · arxiv:1409.2329
    # must be between stacked layers NOT on recurrent connections
    @Rule(Architecture(family=L('lstm') | L('lstm_autoencoder')))
    def dr8_lstm_placement(self):
        self.declare(Blueprint(dropout_placement='between stacked LSTM layers only -- NOT on recurrent connections'))

    # bert
    # Source: Devlin et al. 2018 · arxiv:1810.04805
    # already baked in dont add more
    @Rule(Architecture(family='bert'))
    def dr9_bert_placement(self):
        self.declare(Blueprint(dropout_placement='rate=0.1 already in pretrained model -- do not add more'))

    # gpt
    # Source: Radford et al. 2019
    @Rule(Architecture(family='gpt'))
    def dr10_gpt_placement(self):
        self.declare(Blueprint(dropout_placement='rate=0.1 already in pretrained model -- do not add more'))

    #==============================================
    # transformer_ts
    # Source: Vaswani et al. 2017 · arxiv:1706.03762
    @Rule(Architecture(family='transformer_ts'))
    def dr11_transformer_ts_placement(self):
        self.declare(Blueprint(dropout_placement='Dropout(0.1) in attention layers'))

    # autoencoder
    # Source: Srivastava et al. 2014 · JMLR 15
    @Rule(Architecture(family='autoencoder'))
    def dr12_autoencoder_placement(self):
        self.declare(Blueprint(dropout_placement='in encoder only -- not in decoder'))

    #==============================================
    # xgboost -- no dropout use these params instead
    # Source: Chen & Guestrin 2016 · arxiv:1603.02754
    @Rule(Architecture(family='xgboost'))
    def dr13_xgboost_placement(self):
        self.declare(Blueprint(dropout_placement='not applicable -- use subsample and colsample_bytree instead'))

    # logreg
    # Source: Goodfellow et al. 2016
    @Rule(Architecture(family='logreg'))
    def dr14_logreg_placement(self):
        self.declare(Blueprint(dropout_placement='not applicable -- use L1/L2 regularization instead'))

    # cnn_1d
    # Source: Srivastava et al. 2014 · JMLR 15
    @Rule(Architecture(family='cnn_1d'))
    def dr15_cnn1d_placement(self):
        self.declare(Blueprint(dropout_placement='before final linear layer'))

    #==============================================
    # wav2vec
    # Source: Baevski et al. 2020 · arxiv:2006.11477
    @Rule(Architecture(family='wav2vec'))
    def dr16_wav2vec_placement(self):
        self.declare(Blueprint(dropout_placement='rate=0.1 fixed in pretrained model'))

    # gan
    # Source: Goodfellow et al. 2014 · arxiv:1406.2661
    @Rule(Architecture(family='gan'))
    def dr17_gan_placement(self):
        self.declare(Blueprint(dropout_placement='in discriminator hidden layers only'))