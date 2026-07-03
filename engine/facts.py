from experta import Fact, Field

class Problem(Fact):
    """
    Declared by the CLI from user answers (Q1-Q12).
    These are the inputs to all rules.
    """
    modality        = Field(str)   # 'image' | 'text' | 'tabular' | 'audio' | 'timeseries'
    task            = Field(str)   # 'classify' | 'regress' | 'generate' | 'anomaly' | 'detect'
    output_type     = Field(str)   # 'binary' | 'multiclass' | 'continuous_single' | 'continuous_multi'
    num_classes     = Field(int)
    dataset_size    = Field(str)   # 'tiny' | 'small' | 'medium' | 'large'
    sequential      = Field(bool)
    spatial         = Field(bool)
    is_pretrained   = Field(bool)
    realtime        = Field(bool)
    compute         = Field(str)   # 'cpu' | 'single_gpu' | 'multi_gpu'
    class_imbalance = Field(bool)
    interpretability= Field(bool)
    input_shape     = Field(str)   # e.g. 'small' | 'medium' | 'large'
    outliers_expected  = Field(bool)


class Architecture(Fact):
    """
    Declared by Stage 1 rules (Block A).
    Never set by the user — always derived by the engine.
    """
    family       = Field(str)   # 'mlp' | 'cnn_scratch' | 'cnn_pretrained' |
                                 # 'bert' | 'gpt' | 'lstm' | 'transformer_ts' |
                                 # 'autoencoder' | 'yolo' | 'xgboost' | 'logreg'
    is_pretrained= Field(bool)


class Blueprint(Fact):
    """
    Declared by Stage 2 and 3 rules.
    Each rule adds one Blueprint fact. The formatter collects them all.
    """
    input_layer   = Field(str)
    hidden_layers = Field(str)
    output_layer  = Field(str)
    output_activation = Field(str)
    activation   = Field(str)
    normalization = Field(str)
    dropout         = Field(str)
    loss        = Field(str)
    optimizer      = Field(str)
    lr            = Field(str)
    init         = Field(str)
    notes         = Field(str)
    notes_input    = Field(str)
    width         = Field(str)
    dropout_rate      =Field(str)
    dropout_placement = Field(str)
    lr_schedule = Field(str)
    