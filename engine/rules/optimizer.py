from experta import Rule, L, NOT
from engine.facts import Problem, Architecture, Blueprint


class OptimizerRules:

    # optimizer rules
    # each family hits one optimizer rule

    # Source: Loshchilov & Hutter 2019 · arxiv:1711.05101
    @Rule(Architecture(family=L('bert') | L('gpt')))
    def o1_adamw_transformers(self):
        self.declare(Blueprint(
            optimizer='AdamW',
            lr='2e-5',
            lr_schedule='linear warmup 10% of total steps then cosine decay'
        ))

    #==============================================
    # Source: Loshchilov & Hutter 2019 · arxiv:1711.05101
    @Rule(Architecture(family='transformer_ts'))
    def o2_adamw_ts(self):
        self.declare(Blueprint(
            optimizer='AdamW',
            lr='1e-4',
            lr_schedule='warmup 5% then cosine decay'
        ))

    # cnn scratch
    # Source: Wilson et al. 2017 · arxiv:1705.08292
    # sgd with momentum often converges to better minima than adam for cnns
    @Rule(Architecture(family='cnn_scratch'))
    def o3_sgd_cnn_scratch(self):
        self.declare(Blueprint(
            optimizer='SGD',
            lr='0.01',
            lr_schedule='momentum=0.9, weight_decay=1e-4'
        ))

    #==============================================
    # cnn pretrained
    # Source: Yosinski et al. 2014 · arxiv:1411.1792
    # use lower lr for backbone to avoid destroying pretrained features
    @Rule(Architecture(family='cnn_pretrained'))
    def o4_sgd_cnn_pretrained(self):
        self.declare(Blueprint(
            optimizer='SGD',
            lr='backbone=0.001, head=0.01',
            lr_schedule='momentum=0.9'
        ))

    # conv_autoencoder
    # Source: Kingma & Ba 2014 · arxiv:1412.6980
    @Rule(Architecture(family='conv_autoencoder'))
    def o5_adam_conv_ae(self):
        self.declare(Blueprint(
            optimizer='Adam',
            lr='1e-3',
            lr_schedule='no schedule needed by default'
        ))

    #==============================================
    # mlp
    # Source: Kingma & Ba 2014 · arxiv:1412.6980
    @Rule(Architecture(family='mlp'))
    def o6_adam_mlp(self):
        self.declare(Blueprint(
            optimizer='Adam',
            lr='1e-3',
            lr_schedule='no schedule needed by default'
        ))

    # lstm / lstm_autoencoder
    # Source: Pascanu et al. 2013 · arxiv:1211.5063
    # gradient clipping is essential for lstms to prevent exploding gradients
    @Rule(Architecture(family=L('lstm') | L('lstm_autoencoder')))
    def o7_adam_lstm(self):
        self.declare(Blueprint(
            optimizer='Adam',
            lr='1e-3',
            lr_schedule='gradient clipping max_norm=1.0'
        ))

    #==============================================
    # autoencoder
    # Source: Kingma & Ba 2014 · arxiv:1412.6980
    @Rule(Architecture(family='autoencoder'))
    def o8_adam_autoencoder(self):
        self.declare(Blueprint(
            optimizer='Adam',
            lr='1e-3',
            lr_schedule='no schedule needed by default'
        ))

    # yolo
    # Source: Ultralytics docs · docs.ultralytics.com
    @Rule(Architecture(family='yolo'))
    def o9_sgd_yolo(self):
        self.declare(Blueprint(
            optimizer='SGD',
            lr='0.01',
            lr_schedule='momentum=0.937, weight_decay=5e-4'
        ))

    #==============================================
    # wav2vec
    # Source: Baevski et al. 2020 · arxiv:2006.11477
    @Rule(Architecture(family='wav2vec'))
    def o10_adamw_wav2vec(self):
        self.declare(Blueprint(
            optimizer='AdamW',
            lr='1e-4',
            lr_schedule='warmup 10% then linear decay'
        ))

    # cnn_1d
    # Source: Kingma & Ba 2014 · arxiv:1412.6980
    @Rule(Architecture(family='cnn_1d'))
    def o11_adam_cnn1d(self):
        self.declare(Blueprint(
            optimizer='Adam',
            lr='1e-3',
            lr_schedule='no schedule needed by default'
        ))

    #==============================================
    # xgboost -- no optimizer in the deep learning sense
    # Source: Chen & Guestrin 2016 · arxiv:1603.02754
    @Rule(Architecture(family='xgboost'))
    def o12_xgboost_no_optimizer(self):
        self.declare(Blueprint(
            optimizer='not applicable -- gradient boosting uses internal procedure',
            lr='tune learning_rate=0.01 to 0.3',
            lr_schedule='use early stopping with eval_metric'
        ))

    # logreg
    # Source: Goodfellow et al. 2016
    @Rule(Architecture(family='logreg'))
    def o13_logreg_no_optimizer(self):
        self.declare(Blueprint(
            optimizer='not applicable -- sklearn uses lbfgs or liblinear internally',
            lr='tune C parameter (inverse regularization strength)',
            lr_schedule='not applicable'
        ))

    #==============================================
    # gan
    # Source: Goodfellow et al. 2014 · arxiv:1406.2661
    # both generator and discriminator use adam with specific beta values
    @Rule(Architecture(family='gan'))
    def o14_adam_gan(self):
        self.declare(Blueprint(
            optimizer='Adam for both generator and discriminator',
            lr='2e-4',
            lr_schedule='beta1=0.5, beta2=0.999'
        ))

    # extra notes rules
    # these write to notes, not optimizer, so they do not conflict

    # Source: Loshchilov & Hutter 2016 · arxiv:1608.03983
    # small datasets benefit from warm restarts
    @Rule(
        Problem(dataset_size=L('tiny') | L('small')),
        Architecture(family=L('mlp') | L('cnn_scratch') | L('lstm') | L('autoencoder'))
    )
    def o15_cosine_schedule_small(self):
        self.declare(Blueprint(notes='small dataset detected -- add CosineAnnealingLR or ReduceLROnPlateau'))

    #==============================================
    # pretrained model differential lr note
    # Source: Howard & Ruder 2018 · arxiv:1801.06146
    @Rule(Architecture(is_pretrained=True))
    def o16_differential_lr(self):
        self.declare(Blueprint(notes='pretrained model -- use differential LR: head LR should be 10x the backbone LR'))

    # Source: PyTorch docs · pytorch.org/docs/stable/distributed
    @Rule(Problem(compute='multi_gpu'))
    def o17_multi_gpu_note(self):
        self.declare(Blueprint(notes='multi-GPU detected -- wrap model with torch.nn.parallel.DistributedDataParallel'))

    #==============================================
    # Source: Micikevicius et al. 2017 · arxiv:1710.03740
    # mixed precision cuts memory usage and speeds up training
    @Rule(Problem(realtime=True))
    def o18_realtime_mixed_precision(self):
        self.declare(Blueprint(notes='real-time inference required -- use torch.cuda.amp.autocast for mixed precision'))