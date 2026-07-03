from experta import Rule, L, NOT
from engine.facts import Problem, Architecture, Blueprint


class InitRules:

    #------------------------------------
    # mlp / cnn_scratch
    # Source: He et al. 2015 · arxiv:1502.01852

    @Rule(Architecture(family=L('mlp') | L('cnn_scratch')))
    def i1_he_normal(self):
        self.declare(Blueprint(init='He (Kaiming) Normal -- for ReLU / LeakyReLU layers'))

    # cnn_1d
    # Source: He et al. 2015 · arxiv:1502.01852
    @Rule(Architecture(family='cnn_1d'))
    def i2_he_cnn1d(self):
        self.declare(Blueprint(init='He (Kaiming) Normal for all Conv1D layers'))

    #----------------------------------------------
    # conv_autoencoder
    # Source: He et al. 2015 · arxiv:1502.01852
    @Rule(Architecture(family='conv_autoencoder'))
    def i3_he_conv_ae(self):
        self.declare(Blueprint(init='He (Kaiming) Normal in encoder -- mirror for decoder'))

    # cnn pretrained -- backbone keeps pretrained weights only reinitialize the custom head
    # Source: Kornblith et al. 2019 · arxiv:1805.08974
    @Rule(Architecture(family='cnn_pretrained'))
    def i4_pretrained_cnn(self):
        self.declare(Blueprint(init='pretrained weights for backbone -- He Normal for custom head only'))

    #==============================
    # bert / gpt / wav2vec
    # Source: Devlin et al. 2018 · arxiv:1810.04805
    @Rule(Architecture(family=L('bert') | L('gpt') | L('wav2vec')))
    def i5_pretrained_transformers(self):
        self.declare(Blueprint(init='pretrained weights -- reinitialize classification head only'))

    # transformer_ts
    # Source: Glorot & Bengio 2010 · PMLR v9
    @Rule(Architecture(family='transformer_ts'))
    def i6_transformer_ts_init(self):
        self.declare(Blueprint(init='Xavier Uniform for embedding layers -- He Normal for feedforward layers'))

    #----------------------------------------------
    # lstm / lstm_autoencoder
    # Source: Saxe et al. 2013 · arxiv:1312.6120
    # orthogonal init improves gradient flow in recurrent networks
    @Rule(Architecture(family=L('lstm') | L('lstm_autoencoder')))
    def i7_lstm_init(self):
        self.declare(Blueprint(init='PyTorch default (uniform) -- orthogonal init improves training stability'))

    # autoencoder
    # Source: He et al. 2015 · arxiv:1502.01852
    @Rule(Architecture(family='autoencoder'))
    def i8_autoencoder_init(self):
        self.declare(Blueprint(init='He Normal for encoder -- mirror for decoder'))

    #==============================
    # xgboost / logreg -- no weight init
    # Source: Chen & Guestrin 2016 · arxiv:1603.02754
    @Rule(Architecture(family=L('xgboost') | L('logreg')))
    def i9_no_init(self):
        self.declare(Blueprint(init='not applicable -- tree model / linear model has no weight initialization'))

    # gan
    # Source: Goodfellow et al. 2014 · arxiv:1406.2661
    @Rule(Architecture(family='gan'))
    def i10_gan_init(self):
        self.declare(Blueprint(init='He Normal for both generator and discriminator'))