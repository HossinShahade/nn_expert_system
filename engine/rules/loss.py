from experta import Rule, L, NOT
from engine.facts import Problem, Architecture, Blueprint


class LossRules:

    # output type based rules
    # each output type hits one rule
    # source: PyTorch docs · pytorch.org

    @Rule(
        Problem(output_type='binary', class_imbalance=False),
        NOT(Architecture(family=L('xgboost') | L('logreg') | L('autoencoder') |
                          L('conv_autoencoder') | L('lstm_autoencoder') | L('gpt') |
                          L('yolo') | L('transformer_ts') | L('wav2vec') | L('gan')))
    )
    def l1_bce(self):
        self.declare(Blueprint(loss='BCEWithLogitsLoss'))

    # Source: King & Zeng 2001 · arxiv:cs/0011027
    @Rule(
        Problem(output_type='binary', class_imbalance=True),
        NOT(Architecture(family=L('xgboost') | L('logreg') | L('autoencoder') |
                          L('conv_autoencoder') | L('lstm_autoencoder') | L('gpt') |
                          L('yolo') | L('transformer_ts') | L('wav2vec') | L('gan')))
    )
    def l2_bce_weighted(self):
        self.declare(Blueprint(loss='BCEWithLogitsLoss with pos_weight = num_negatives / num_positives'))

    @Rule(
        Problem(output_type='multiclass', class_imbalance=False),
        NOT(Architecture(family=L('xgboost') | L('logreg') | L('autoencoder') |
                          L('conv_autoencoder') | L('lstm_autoencoder') | L('gpt') |
                          L('yolo') | L('transformer_ts') | L('wav2vec') | L('gan')))
    )
    def l3_crossentropy(self):
        self.declare(Blueprint(loss='CrossEntropyLoss'))

    #==============================================
    # focal loss for class imbalance
    # Source: Lin et al. 2017 · arxiv:1708.02002
    @Rule(
        Problem(output_type='multiclass', class_imbalance=True),
        NOT(Architecture(family=L('xgboost') | L('logreg') | L('autoencoder') |
                          L('conv_autoencoder') | L('lstm_autoencoder') | L('gpt') |
                          L('yolo') | L('transformer_ts') | L('wav2vec') | L('gan')))
    )
    def l4_focal(self):
        self.declare(Blueprint(loss='FocalLoss (gamma=2, alpha=0.25)'))

    @Rule(
        Problem(output_type='continuous_single'),
        NOT(Architecture(family=L('xgboost') | L('logreg') | L('autoencoder') |
                          L('conv_autoencoder') | L('lstm_autoencoder') | L('gpt') |
                          L('yolo') | L('transformer_ts') | L('wav2vec') | L('gan')))
    )
    def l5_mse(self):
        self.declare(Blueprint(loss='MSELoss'))

    @Rule(
        Problem(output_type='continuous_multi'),
        NOT(Architecture(family=L('xgboost') | L('logreg') | L('autoencoder') |
                          L('conv_autoencoder') | L('lstm_autoencoder') | L('gpt') |
                          L('yolo') | L('transformer_ts') | L('wav2vec') | L('gan')))
    )
    def l6_mse_multi(self):
        self.declare(Blueprint(loss='MSELoss across all output dimensions'))

    # architecture based rules
    # these override or supplement the output type rules above
    # higher salience means they take priority

    #==============================================
    # autoencoders use reconstruction loss regardless of output type
    # Source: Kingma & Welling 2013 · arxiv:1312.6114
    @Rule(Architecture(family=L('autoencoder') | L('conv_autoencoder') | L('lstm_autoencoder')), salience=20)
    def l7_autoencoder_reconstruction(self):
        self.declare(Blueprint(loss='MSELoss (reconstruction) -- add KL divergence if using VAE variant'))

    #==============================================
    # gpt generation uses label smoothing
    # Source: Muller et al. 2019 · arxiv:1906.02629
    @Rule(Architecture(family='gpt'), salience=20)
    def l8_gpt_label_smooth(self):
        self.declare(Blueprint(loss='CrossEntropyLoss with label smoothing epsilon=0.1'))

    # yolo has a composite loss
    # Source: Lin et al. 2017 · arxiv:1708.02002
    @Rule(Architecture(family='yolo'), salience=20)
    def l9_yolo_loss(self):
        self.declare(Blueprint(loss='FocalLoss (objectness) + CIoU loss (box regression) + CrossEntropy (class)'))

    #==============================================
    # Source: Chen & Guestrin 2016 · arxiv:1603.02754
    @Rule(Architecture(family='xgboost'), salience=20)
    def l10_xgboost_loss(self):
        self.declare(Blueprint(loss='handled internally -- set eval_metric=logloss (classify) or rmse (regress)'))

    # Source: Goodfellow et al. 2016
    @Rule(Architecture(family='logreg'), salience=20)
    def l11_logreg_loss(self):
        self.declare(Blueprint(loss='log-loss -- handled internally by sklearn'))

    #==============================================
    # transformer_ts forecasting
    # Source: Zhou et al. 2021 · arxiv:2012.07436
    @Rule(Architecture(family='transformer_ts'), salience=20)
    def l12_transformer_ts_loss(self):
        self.declare(Blueprint(loss='MSELoss on forecast output'))

    # Source: Baevski et al. 2020 · arxiv:2006.11477
    @Rule(Architecture(family='wav2vec'), salience=20)
    def l13_wav2vec_loss(self):
        self.declare(Blueprint(loss='CrossEntropyLoss on pooled output'))

    #==============================================
    # gan uses adversarial loss
    # Source: Goodfellow et al. 2014 · arxiv:1406.2661
    @Rule(Architecture(family='gan'), salience=20)
    def l14_gan_loss(self):
        self.declare(Blueprint(loss='Generator: BCE or Wasserstein loss -- Discriminator: BCE'))