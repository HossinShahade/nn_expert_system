from experta import Rule,L,salience
from engine.facts import Problem,Architecture,Blueprint




class DepthRules:
    #MLP family
    #source Raghu et al . 2017 arxiv:1606.05336
    @Rule(Architecture(family = 'mlp'),Problem(dataset_size ='tiny'))
    def mlp_depth_tiny(self):
        self.declare(Blueprint(hidden_layers='1 hidden laayer'))
    @ Rule (Architecture(family='mlp'),Problem(dataset_size='small'))
    def mlp_depth_small(self):
        self.declare(Blueprint(hidden_layers='2 hidden layers'))    
    @Rule(Architecture(family='mlp'),Problem(dataset_size='medium'))
    def mlp_depth_medium(self):
        self.declare(Blueprint(hidden_layers = ' 3 to 4 hidden layers'))
    @Rule(Architecture(family = 'mlp'),Problem(dataset_size='large'))
    def mlp_depth_larg(self):
        self.declare(Blueprint(hidden_layers='4 to 5 hidden layers'))    
    
    
    #------------------------------------=---
    # CNN scratch depth rules
    # Source: He et al. 2015 - arxiv:1512.03385
    # Source: Simonyan & Zisserman 2014 - arxiv:1409.1556

    @Rule (Architecture(family ='cnn_scratch'),Problem(dataset_size=L('tiny')| L('small')))
    def cnn_depth_tiny(self):
        self.declare(Blueprint(hidden_layers='3  to 4 conv blocks'))
    @Rule (Architecture(family = 'cnn_scratch'),Problem(dataset_size='medium'))
    def cnn_depth_medium(self):
        self.declare(Blueprint(hidden_layers = '5 to 7 conv blocks'))
    @Rule (Architecture(family ='cnn_scratch'),Problem(dataset_size='large'))
    def cnn_depth_larg(self):
        self.declare(Blueprint(hidden_layers ='8 to 12 conv blocks plus residuals'))
    
    # pretained cnn
    #Kornblith et al. 2019 · arxiv:1805.08974
    @Rule(Architecture(family='cnn_pretrained'),salience=20)
    def cnn_pretrained (self):
        self.declare(Blueprint(hidden_layers='fixed by pretrained backbone',notes='only classification depth is user_set(1-2FC layers)'))
        
    #____________________________________
    #conv_autocoder
    #Gong et al. 2019 · arxiv:1904.11294
    @Rule(Architecture(family='conv_autoencoder'),Problem(dataset_size=L('tiny')|L('small')),salience=10)
    def conv_autoencoder_depth_small(self):
        self.declare(Blueprint(hidden_layers='encoder 2 to 3 conv blocks -> bottel neck -> decoder mirror'),salience=10)
    @Rule(Architecture(family='conv_autoencoder'),Problem(dataset_size=L('medium')|L('large')))
    def conv_autoencoder_depth_larg (self):
        self.declare(Blueprint(hidden_layers='encoder 4 to 5 conv blocks -> bottelneck _. decoder:mirror'))
    
    
    
        
    #E--------------------------------
    #LSTM family
    #   Source: Graves et al. 2013 - arxiv:1303.5778
    @Rule(Architecture(family='lstm'),Problem(dataset_size =L('tiny')|L('small')),salience=10)
    def lstm_depth_small(self):
        self.declare(Blueprint(hidden_layers ='1 LSTM layer'))
    @Rule(Architecture(family='lstm'),Problem(dataset_size='medium'),salience=10)
    def lstm_depth_medium(self):
        self.declare(Blueprint(hidden_layers='2 stacked LSTM layers'))
    @Rule (Architecture(family='lstm'),Problem(dataset_size ='large'),salience=10)
    def lstm_depth_larg(self):
        self.declare(Blueprint(hidden_layers='3 to 4 stacked lstm layers'))
    
    #------------------
    #autoencoder family
    #Source: Hinton & Salakhutdinov 2006
    
    @Rule (Architecture(family='autoencoder'),salience=10)
    def autoencoder_depth(self):
        self.declare(Blueprint(hidden_layers='encode:3 to 4 layers -> Bottelneck -> decoder:mirror encoder'))
    
    #---------------
    #cpu
    #overrid the perivios depth
    # Source: Howard et al. 2017 - arxiv:1704.04861
    @Rule(Problem(compute='cpu'),Architecture(family= L('mlp')| L('cnn_scratch')),salience=30)
    def cpu_depth_cap(self):
        self.declare(Blueprint(notes="depth capped at the compute level of the cpu"))
    
    
    
    #_-----------------------------------
    #lstm autoencoder
    #Malhotra et al. 2016 · arxiv:1607.00148
    @Rule(Architecture(family='lstm_autoencoder'),Problem(dataset_size=L('tiny')|L('small')),salience=10)
    def lstm_autoencoder_depth_small(self):
        self.declare(Blueprint(hidden_layers='encoder 1 lstm layer->bottelneck_>decoder mirror'))
    @Rule(Architecture(family='lstm_autoencoder'),Problem(dataset_size=L('medium')|L('large')),salience=10)
    def lstm_autoencoder_depth_larg(self):
        self.declare(Blueprint(hiddenlayers='encoder 2 lstm layers -> bottel neck _. decoder mirror'))
    
    
    
    #____________________________________________________--
    #bret family 
    #Devlin et al. 2018 · arxiv:1810.04805 
    
    
    @Rule (Architecture(family='bert'),salience=20)
    def bert_depth(self):
        self.declare(Blueprint(hidden_layers='12 transformer layers (bert base) or 24 (bert base) fixed by pretrained model'))
    
    
    
    #________________________________________________________
    #gpt family
    #Radford et al. 2019
    
    
    
    @Rule(Architecture(family='gpt'),salience=20)
    def gpt_depth(self):
        self.declare(Blueprint(hidden_layers='depth fixed by model variant ex: gpt 2 small 12,medium 24 larg 36'))
        
    
    
    
    #_________________________________________________
    #transformers
    #Zhou et al. 2021 · arxiv:2012.07436
    @Rule(Architecture(family='transformer_ts'),Problem(dataset_size=L('tiny')|L('small')),salience=10)
    def transformer_ts_depth_small(self):
        self.declare(Blueprint(hidden_layers='2 encoder layers'))
        
    @Rule(Architecture(family='transformer_ts'),Problem(dataset_size='medium'),salience=10)
    def transformer_ts_depth_medium(self):
        self.declare(Blueprint(hidden_layers='4 encoder layers'))
    @Rule(Architecture(family='transformer_ts'),Problem(dataset_size='larg'),salience=10)
    def transformer_ts_depth_larg(self):
        self.declare(Blueprint(hidden_layers='6 encoder layyers'))
    #_____________________________________
    #xgboost
    #Chen & Guestrin 2016 · arxiv:1603.02754
    
    @Rule(Architecture(family='xgboost'),salience=10)
    def xgboost_depth(self):
        self.declare(Blueprint(hidden_layers='n-estimators=100,300 for small data or 300-500 for the larg data'))
    #__________________________________________
    #logreg
    # Goodfellow et al. 2016
    @Rule(Architecture(family='logreg'),salience=10)
    def logreg_depth(self):
        self.declare(Blueprint(hidden_layers='singel layer by diffinition no hidden layers or depth'))
    #____________________________________
    #yolo
    #Ultralytics docs · docs.ultralytics.com
    @Rule(Architecture(family='yolo'),salience=10)
    def yolo_depth(self):
        self.declare(Blueprint(hidden_layers='depth fixed by model variant ex: YOLOv8n/s/m/1/x'))
    #_________________________________________
    #wav2vec
    #Baevski et al. 2020 · arxiv:2006.11477
    @Rule(Architecture(family='wav2vec'),salience=10)
    def wav2vec_depth(self):
        self.declare(Blueprint(hidden_layers='depth fixed by pretrained model 12 or 24 transformer layers'))
        
    #__________________________________________________________
    #cnn_1d
    #LeCun et al. 1995
    @Rule(Architecture(family='cnn_1d'),Problem(dataset_size= L('tiny')|L('small')),salience=10)
    def cnn_1d_depth_small(self):
        self.declare(Blueprint(hidden_layers='3 conv layers'))
    @Rule(Architecture(family='cnn_1d'),Problem(dataset_size= L('medium')|L('larg')),salience=10)
    def cnn_1_d_depth_larg(self):
        self.declare(Blueprint(hidden_layers='4 to 5 conv layrs'))
    #___________________
    #realtime bottelneck
    #Howard et al. 2017 · arxiv:1704.04861
    @Rule(Problem(realtime=True),salience=25)
    def realtime_cap_depth(self):
        self.declare(Blueprint(notes='depth capped to minimum values'))