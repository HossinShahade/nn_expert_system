from experta import Rule, L, NOT
from engine.facts import Problem, Architecture, Blueprint


class WidthRules:

    #____________________________________
    # mlp cpu
    # Goodfellow et al. 2016

    @Rule(Architecture(family='mlp'), Problem(compute='cpu'), salience=25)
    def mlp_cpu_width(self):
        self.declare(Blueprint(width='min(128, 2X input features) -> halve each layer'))

    # mlp single gpu
    @Rule(Architecture(family='mlp'), Problem(compute='single_gpu'), salience=15)
    def mlp_single_gpu_width(self):
        self.declare(Blueprint(width='min(512, 4X input features) -> halve each layer'))

    # mlp multi gpu
    @Rule(Architecture(family='mlp'), Problem(compute='multi_gpu'), salience=10)
    def mlp_multi_gpu_width(self):
        self.declare(Blueprint(width='min(1024, 8X input features) -> halve each layer'))

    #_______________________________________
    # cnn scratch
    # Simonyan & Zisserman 2014 · arxiv:1409.1556

    # cnn scratch cpu
    @Rule(Architecture(family='cnn_scratch'), Problem(compute='cpu'), salience=25)
    def cnn_scratch_cpu_width(self):
        self.declare(Blueprint(width='16 filters -> double each block, cap at 128'))

    # cnn scratch single gpu
    @Rule(Architecture(family='cnn_scratch'), Problem(compute='single_gpu'), salience=15)
    def cnn_scratch_single_gpu_width(self):
        self.declare(Blueprint(width='32 filters -> double each block, cap at 512'))

    # cnn scratch multi gpu
    @Rule(Architecture(family='cnn_scratch'), Problem(compute='multi_gpu'), salience=10)
    def cnn_scratch_multi_gpu_width(self):
        self.declare(Blueprint(width='64 filters -> double each block, cap at 1024'))

    #__________________________________________
    # cnn pretrained
    # Kornblith et al. 2019 · arxiv:1805.08974

    @Rule(Architecture(family='cnn_pretrained'), salience=10)
    def cnn_pretrained_width(self):
        self.declare(Blueprint(width='head: linear(backbone_out->256) -> linear(256->number of classes)'))

    #_________________________
    # conv_autoencoder
    # Gong et al. 2019 · arxiv:1904.11294

    @Rule(Architecture(family='conv_autoencoder'), Problem(compute='cpu'), salience=25)
    def conv_autoencoder_cpu_width(self):
        self.declare(Blueprint(width='16->32->bottelneck(8)->32->16'))

    @Rule(Architecture(family='conv_autoencoder'), Problem(compute='single_gpu'), salience=15)
    def conv_autoencoder_single_gpu_width(self):
        self.declare(Blueprint(width='32->64->bottelneck(16)->64->32'))

    @Rule(Architecture(family='conv_autoencoder'), Problem(compute='multi_gpu'), salience=10)
    def conv_autoencoder_multi_gpu_width(self):
        self.declare(Blueprint(width='64->128->bottelneck(32)->128->64'))

    #_______________________________________________
    # lstm
    # Graves et al. 2013 · arxiv:1303.5778

    # lstm cpu
    @Rule(Architecture(family='lstm'), Problem(compute='cpu'), salience=25)
    def lstm_cpu_width(self):
        self.declare(Blueprint(width='hidden size=64'))

    # lstm single gpu
    @Rule(Architecture(family='lstm'), Problem(compute='single_gpu'), salience=15)
    def lstm_single_gpu_width(self):
        self.declare(Blueprint(width='hidden size=128 to 256'))

    # lstm multi gpu
    @Rule(Architecture(family='lstm'), Problem(compute='multi_gpu'), salience=10)
    def lstm_multi_gpu_width(self):
        self.declare(Blueprint(width='hidden size=256 to 512'))

    # lstm classify note -- writes to notes not width to avoid conflict
    # Graves & Schmidhuber 2005
    @Rule(Architecture(family='lstm'), Problem(task='classify'), salience=20)
    def lstm_classify_width(self):
        self.declare(Blueprint(notes='use Bidirectional LSTM -- doubles the effective hidden size'))

    #_________________________________________
    # lstm_autoencoder
    # Malhotra et al. 2016 · arxiv:1607.00148

    @Rule(Architecture(family='lstm_autoencoder'), Problem(compute='cpu'), salience=25)
    def lstm_autoencoder_cpu_width(self):
        self.declare(Blueprint(width='hidden size=32 encoder, mirror decoder'))

    @Rule(Architecture(family='lstm_autoencoder'), Problem(compute='single_gpu'), salience=15)
    def lstm_autoencoder_single_gpu_width(self):
        self.declare(Blueprint(width='hidden size=64 to 128 encoder, mirror decoder'))

    @Rule(Architecture(family='lstm_autoencoder'), Problem(compute='multi_gpu'), salience=10)
    def lstm_autoencoder_multi_gpu_width(self):
        self.declare(Blueprint(width='hidden size=128 to 256 encoder, mirror decoder'))

    #_______________________________
    # bert
    # Devlin et al. 2018 · arxiv:1810.04805

    @Rule(Architecture(family='bert'), salience=20)
    def bert_width(self):
        self.declare(Blueprint(width='hidden size=768 (BERT-base) or 1024 (BERT-large) -- fixed'))

    # bert head
    # Sun et al. 2019 · arxiv:1905.05583
    @Rule(Architecture(family='bert'), salience=10)
    def bert_head_width(self):
        self.declare(Blueprint(notes='head: linear(768->256) -> linear(256->num_classes)'))

    #------------------------
    # gpt
    # Radford et al. 2019

    @Rule(Architecture(family='gpt'), salience=20)
    def gpt_width(self):
        self.declare(Blueprint(width='hidden size fixed by model variant -- not user-set'))

    #_________________
    # transformer_ts
    # Zhou et al. 2021 · arxiv:2012.07436

    @Rule(Architecture(family='transformer_ts'), Problem(compute='cpu'), salience=25)
    def transformer_ts_cpu_width(self):
        self.declare(Blueprint(width='model_dim=64, num_heads=4'))

    @Rule(Architecture(family='transformer_ts'), Problem(compute='single_gpu'), salience=15)
    def transformer_ts_single_gpu_width(self):
        self.declare(Blueprint(width='model_dim=128, num_heads=8'))

    @Rule(Architecture(family='transformer_ts'), Problem(compute='multi_gpu'), salience=10)
    def transformer_ts_multi_gpu_width(self):
        self.declare(Blueprint(width='model_dim=256, num_heads=8'))

    #________________________________
    # autoencoder
    # Hinton & Salakhutdinov 2006 · Bank et al. 2023

    @Rule(Architecture(family='autoencoder'), salience=10)
    def autoencoder_width(self):
        self.declare(Blueprint(width='bottelneck = input_dim/4 or /8 for aggressive compression'))

    #______________________________
    # xgboost
    # Chen & Guestrin 2016 · arxiv:1603.02754

    @Rule(Architecture(family='xgboost'), salience=10)
    def xgboost_width(self):
        self.declare(Blueprint(width='max_depth=3 to 6, controls tree width not layer width'))

    #------------------------------
    # logreg
    # Goodfellow et al. 2016

    @Rule(Architecture(family='logreg'), salience=10)
    def logreg_width(self):
        self.declare(Blueprint(width='no width, singel layer, width=number of classes'))

    #------------------------------
    # cnn_1d
    # LeCun et al. 1995

    @Rule(Architecture(family='cnn_1d'), Problem(compute='cpu'), salience=25)
    def cnn_1d_cpu_width(self):
        self.declare(Blueprint(width='16 filters->32->64'))

    @Rule(Architecture(family='cnn_1d'), Problem(compute='single_gpu'), salience=15)
    def cnn_1d_single_gpu_width(self):
        self.declare(Blueprint(width='32 filters->64->128'))

    @Rule(Architecture(family='cnn_1d'), Problem(compute='multi_gpu'), salience=10)
    def cnn_1d_multi_gpu_width(self):
        self.declare(Blueprint(width='64 filters->128->256'))

    #------------------------------
    # gan
    # Goodfellow et al. 2014 · arxiv:1406.2661

    @Rule(Architecture(family='gan'), salience=10)
    def gan_width(self):
        self.declare(Blueprint(width='Generator: 128->256->512->1024 -- Discriminator: 1024->512->256->1'))

    #------------------------------
    # wav2vec
    # Baevski et al. 2020 · arxiv:2006.11477

    @Rule(Architecture(family='wav2vec'), salience=10)
    def wav2vec_width(self):
        self.declare(Blueprint(width='hidden size fixed by pretrained model -- not user-set'))