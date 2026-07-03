from experta import Rule, L, NOT
from engine.facts import Problem, Architecture, Blueprint


class NormalizationRules:

    #_________________________________-
    # mlp
    # Source: Ioffe & Szegedy 2015 · arxiv:1502.03167
    # batchnorm works well when you have enough samples

    @Rule(Architecture(family='mlp'), Problem(dataset_size=L('small') | L('medium') | L('large')))
    def n1_mlp_batchnorm(self):
        self.declare(Blueprint(normalization='BatchNorm after each linear layer, before activation'))

    # tiny datasets have unreliable batch statistics skip batchnorm
    @Rule(Architecture(family='mlp'), Problem(dataset_size='tiny'))
    def n2_mlp_tiny_no_batchnorm(self):
        self.declare(Blueprint(normalization='Skip BatchNorm -- dataset too small for reliable batch statistics'))

    #__________________________________
    # cnn scratch
    # Source: Ioffe & Szegedy 2015 · arxiv:1502.03167

    @Rule(Architecture(family='cnn_scratch'), Problem(dataset_size=L('small') | L('medium') | L('large')))
    def n3_cnn_batchnorm(self):
        self.declare(Blueprint(normalization='BatchNorm after each conv layer, before activation'))

    # Source: Wu & He 2018 · arxiv:1803.08494
    # groupnorm works better than batchnorm on tiny datasets
    @Rule(Architecture(family='cnn_scratch'), Problem(dataset_size='tiny'))
    def n4_cnn_groupnorm(self):
        self.declare(Blueprint(normalization='GroupNorm(num_groups=8) -- use instead of BatchNorm on tiny datasets'))

    #==============================================
    # cnn pretrained
    # Source: Kornblith et al. 2019 · arxiv:1805.08974
    @Rule(Architecture(family='cnn_pretrained'))
    def n5_cnn_pretrained_norm(self):
        self.declare(Blueprint(normalization='normalization fixed in pretrained backbone -- apply only in custom head'))

    # conv_autoencoder
    # Source: Ioffe & Szegedy 2015 · arxiv:1502.03167
    @Rule(Architecture(family='conv_autoencoder'))
    def n6_conv_autoencoder_norm(self):
        self.declare(Blueprint(normalization='BatchNorm after each conv layer in encoder and decoder'))

    #==============================================
    # bert / gpt
    # Source: Ba et al. 2016 · arxiv:1607.06450
    # layernorm is fixed inside pretrained transformers
    @Rule(Architecture(family=L('bert') | L('gpt')))
    def n7_bert_gpt_norm(self):
        self.declare(Blueprint(normalization='LayerNorm -- fixed in pretrained model, do not modify'))

    # transformer_ts
    # Source: Ba et al. 2016 · arxiv:1607.06450
    @Rule(Architecture(family='transformer_ts'))
    def n8_transformer_ts_norm(self):
        self.declare(Blueprint(normalization='LayerNorm after each attention and feedforward sublayer'))

    #==============================================
    # lstm
    # Source: Ba et al. 2016 · arxiv:1607.06450
    # layernorm between stacked lstm layers is optional but helps
    @Rule(Architecture(family='lstm'))
    def n9_lstm_norm(self):
        self.declare(Blueprint(normalization='LayerNorm between stacked LSTM layers (optional but helps on small data)'))

    @Rule(Architecture(family='lstm_autoencoder'))
    def n10_lstm_autoencoder_norm(self):
        self.declare(Blueprint(normalization='LayerNorm between encoder and decoder LSTM layers'))

    #==============================================
    # autoencoder
    # Source: Ioffe & Szegedy 2015 · arxiv:1502.03167

    @Rule(Architecture(family='autoencoder'), Problem(dataset_size=L('small') | L('medium') | L('large')))
    def n11_autoencoder_batchnorm(self):
        self.declare(Blueprint(normalization='BatchNorm after each linear layer'))

    @Rule(Architecture(family='autoencoder'), Problem(dataset_size='tiny'))
    def n12_autoencoder_tiny(self):
        self.declare(Blueprint(normalization='Skip BatchNorm -- dataset too small'))

    #==============================================
    # xgboost
    # Source: Chen & Guestrin 2016 · arxiv:1603.02754
    # tree models are scale invariant
    @Rule(Architecture(family='xgboost'))
    def n13_xgboost_norm(self):
        self.declare(Blueprint(normalization='not applicable -- tree models are scale-invariant'))

    # logreg
    # Source: Goodfellow et al. 2016
    # no internal normalization but input features need to be scaled
    @Rule(Architecture(family='logreg'))
    def n14_logreg_norm(self):
        self.declare(Blueprint(normalization='no internal normalization -- apply StandardScaler to input features before fitting'))

    #==============================================
    # cnn_1d
    # Source: Ioffe & Szegedy 2015 · arxiv:1502.03167
    @Rule(Architecture(family='cnn_1d'))
    def n15_cnn1d_norm(self):
        self.declare(Blueprint(normalization='BatchNorm after each Conv1D layer'))

    # wav2vec
    # Source: Baevski et al. 2020 · arxiv:2006.11477
    @Rule(Architecture(family='wav2vec'))
    def n16_wav2vec_norm(self):
        self.declare(Blueprint(normalization='LayerNorm -- fixed in pretrained model, do not modify'))

    # gan
    # Source: Goodfellow et al. 2014 · arxiv:1406.2661
    @Rule(Architecture(family='gan'))
    def n17_gan_norm(self):
        self.declare(Blueprint(normalization='BatchNorm in generator -- no norm in discriminator (training stability)'))