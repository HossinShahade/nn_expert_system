from experta import Rule, L, NOT
from engine.facts import Problem, Architecture, Blueprint


class ActivationRules:

    # hidden layer activations
    # these fire based on architecture family only
    # each family matches one rule
    # pretty much one rule per family

    # rule a1
    # source: Nair & Hinton 2010 - ICML 2010
    # relu is the default for mlp and cnn on medium/large data
    @Rule(
        Architecture(family=L('mlp') | L('cnn_scratch')),
        Problem(dataset_size=L('medium') | L('large'))
    )
    def a1_relu_mlp_cnn(self):
        self.declare(Blueprint(activation='ReLU'))

    # Rule A2
    # Source: Maas et al. 2013 - ICML 2013
    # small datasets risk dead neurons so we suggest leakyrelu
    @Rule(
        Architecture(family=L('mlp') | L('cnn_scratch')),
        Problem(dataset_size=L('tiny') | L('small'))
    )
    def a2_leakyrelu_small(self):
        self.declare(Blueprint(activation='ReLU -- consider LeakyReLU (alpha=0.01) if dead neurons observed'))

    #==============================================
    # cnn_pretrained backbone is fixed
    # Source: Kornblith et al. 2019 - arxiv:1805.08974
    @Rule(
        Architecture(family='cnn_pretrained')
    )
    def a3_cnn_pretrained(self):
        self.declare(Blueprint(activation='Fixed by pretrained backbone -- ReLU in custom head layers only'))

    #==============================================
    # bert / gpt / wav2vec
    # Source: Hendrycks & Gimpel 2016 - arxiv:1606.08415
    @Rule(
        Architecture(family=L('bert') | L('gpt') | L('wav2vec'))
    )
    def a4_gelu_pretrained(self):
        self.declare(Blueprint(activation='GELU -- fixed in pretrained model, do not change'))

    #==============================================
    # transformer_ts
    # Source: Hendrycks & Gimpel 2016 - arxiv:1606.08415
    @Rule(
        Architecture(family='transformer_ts')
    )
    def a5_transformer_ts(self):
        self.declare(Blueprint(activation='GELU in feedforward sublayers of each encoder block'))

    #==============================================
    # lstm / lstm_autoencoder
    # Source: Hochreiter & Schmidhuber 1997
    # sigmoid and tanh are built into pytorchs lstm cell
    @Rule(
        Architecture(family=L('lstm') | L('lstm_autoencoder'))
    )
    def a6_lstm(self):
        self.declare(Blueprint(activation='Sigmoid (gates) + Tanh (cell state) -- built into PyTorch LSTM, not user-set'))

    #==============================================
    # autoencoder / conv_autoencoder
    # Source: Bank et al. 2023 - arxiv:2003.05991
    @Rule(
        Architecture(family=L('autoencoder') | L('conv_autoencoder'))
    )
    def a7_autoencoder(self):
        self.declare(Blueprint(activation='ReLU in encoder and decoder hidden layers'))

    #==============================================
    # yolo
    # Source: Ramachandran et al. 2017 - arxiv:1710.05941
    @Rule(
        Architecture(family='yolo')
    )
    def a8_yolo(self):
        self.declare(Blueprint(activation='SiLU (Swish) in backbone -- LeakyReLU in older YOLO versions'))

    #==============================================
    # cnn_1d
    # Source: Nair & Hinton 2010 - ICML 2010
    @Rule(
        Architecture(family='cnn_1d')
    )
    def a9_cnn1d(self):
        self.declare(Blueprint(activation='ReLU after every Conv1D layer'))

    #==============================================
    # xgboost / logreg
    # Source: Chen & Guestrin 2016 - arxiv:1603.02754
    @Rule(
        Architecture(family=L('xgboost') | L('logreg'))
    )
    def a10_no_activation(self):
        self.declare(Blueprint(activation='Not applicable -- tree model / linear model has no activation functions'))

    #==============================================
    # gan generator and discriminator are different parts
    # both rules fire together which is correct
    # Source: Goodfellow et al. 2014 - arxiv:1406.2661
    @Rule(
        Architecture(family='gan')
    )
    def a11_gan(self):
        self.declare(Blueprint(
            activation='Generator: ReLU hidden, Tanh output -- Discriminator: LeakyReLU (alpha=0.2) hidden, Sigmoid output'
        ))

    # final decoder layer for autoencoders
    # separate from hidden activation, this is the reconstruction output
    # source: Bank et al. 2023 - arxiv:2003.05991
    @Rule(
        Architecture(family=L('autoencoder') | L('conv_autoencoder') | L('lstm_autoencoder'))
    )
    def a12_autoencoder_final_layer(self):
        self.declare(Blueprint(
            output_activation='Sigmoid if input normalized to [0,1] -- Tanh otherwise'
        ))

    # output layer activations
    # these write to output_activation, not activation
    # each output type matches one rule
    # source: PyTorch docs - pytorch.org

    # Rule A13
    @Rule(Problem(output_type='binary'))
    def a13_output_binary(self):
        self.declare(Blueprint(
            output_activation='None -- BCEWithLogitsLoss applies Sigmoid internally'
        ))

    # Rule A14
    @Rule(Problem(output_type='multiclass'))
    def a14_output_multiclass(self):
        self.declare(Blueprint(
            output_activation='None -- CrossEntropyLoss applies Softmax internally'
        ))

    # Rule A15
    @Rule(Problem(output_type=L('continuous_single') | L('continuous_multi')))
    def a15_output_regression(self):
        self.declare(Blueprint(
            output_activation='None -- linear output, no activation needed'
        ))