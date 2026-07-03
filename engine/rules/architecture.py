from experta import Rule, NOT, OR, L
from engine.facts import Problem, Architecture, Blueprint


class ArchitectureRules:

    # ── Image classification ─────────────────────────────────────────────

    # Rule AF1
    # Source: Kornblith et al. 2019 - arxiv:1805.08974
    # Pretrained models almost always outperform training from scratch
    @Rule(
        Problem(modality='image', task='classify', is_pretrained=True),
        NOT(Architecture())
    )
    def af1_image_classify_pretrained(self):
        self.declare(Architecture(family='cnn_pretrained', is_pretrained=True))

    # Rule AF2
    # Source: He et al. 2015 - arxiv:1512.03385
    # Large datasets make from-scratch training viable
    @Rule(
        Problem(modality='image', task='classify', is_pretrained=False,
                dataset_size=L('large')),
        NOT(Architecture())
    )
    def af2_image_classify_scratch_large(self):
        self.declare(Architecture(family='cnn_scratch', is_pretrained=False))

    # Rule AF3
    # Source: Kornblith et al. 2019
    # Force pretrained even if user said no — not enough data to train from scratch
    @Rule(
        Problem(modality='image', task='classify', is_pretrained=False,
                dataset_size=L('tiny') | L('small') | L('medium')),
        NOT(Architecture())
    )
    def af3_image_classify_force_pretrained(self):
        self.declare(Architecture(family='cnn_pretrained', is_pretrained=True))
        self.declare(Blueprint(
            notes='WARNING: Dataset too small to train from scratch. '
                  'Forcing pretrained model. (Kornblith et al. 2019)'
        ))

    # Rule AF4
    # Source: Redmon et al. 2016 - arxiv:1506.02640
    @Rule(
        Problem(modality='image', task='detect'),
        NOT(Architecture())
    )
    def af4_object_detection(self):
        self.declare(Architecture(family='yolo', is_pretrained=True))

    # Rule AF5
    # Source: He et al. 2015 - arxiv:1512.03385
    # Regression on images still uses a CNN backbone, just swaps the head
    @Rule(
        Problem(modality='image', task='regress'),
        NOT(Architecture())
    )
    def af5_image_regress(self):
        self.declare(Architecture(family='cnn_pretrained', is_pretrained=True))
        self.declare(Blueprint(
            notes='Swap classification head for a regression head '
                  '(He et al. 2015)'
        ))

    # Rule AF6
    # Source: Gong et al. 2019 - arxiv:1904.11294
    @Rule(
        Problem(modality='image', task='anomaly'),
        NOT(Architecture())
    )
    def af6_image_anomaly(self):
        self.declare(Architecture(family='conv_autoencoder', is_pretrained=False))

    # Rule AF7
    # Source: Goodfellow et al. 2014 - arxiv:1406.2661
    @Rule(
        Problem(modality='image', task='generate'),
        NOT(Architecture())
    )
    def af7_image_generate(self):
        self.declare(Architecture(family='gan', is_pretrained=False))
        self.declare(Blueprint(
            notes='Consider diffusion models for higher generation quality '
                  '(Goodfellow et al. 2014)'
        ))

    # ── Text ─────────────────────────────────────────────────────────────

    # Rule AF8
    # Source: Devlin et al. 2018 - arxiv:1810.04805
    @Rule(
        Problem(modality='text', task='classify', is_pretrained=True),
        NOT(Architecture())
    )
    def af8_text_classify_pretrained(self):
        self.declare(Architecture(family='bert', is_pretrained=True))

    # Rule AF9
    # Source: Graves et al. 2013 - arxiv:1303.5778
    @Rule(
        Problem(modality='text', task='classify', is_pretrained=False),
        NOT(Architecture())
    )
    def af9_text_classify_no_pretrained(self):
        self.declare(Architecture(family='lstm', is_pretrained=False))

    # Rule AF10
    # Source: Devlin et al. 2018 - arxiv:1810.04805
    @Rule(
        Problem(modality='text', task='regress', is_pretrained=True),
        NOT(Architecture())
    )
    def af10_text_regress_pretrained(self):
        self.declare(Architecture(family='bert', is_pretrained=True))
        self.declare(Blueprint(
            notes='Use a regression head on the [CLS] token '
                  '(Devlin et al. 2018)'
        ))

    # Rule AF11
    # Source: Raffel et al. 2019 - arxiv:1910.10683
    @Rule(
        Problem(modality='text', task='generate'),
        NOT(Architecture())
    )
    def af11_text_generate(self):
        self.declare(Architecture(family='gpt', is_pretrained=True))

    # Rule AF12
    # Source: Malhotra et al. 2016 - arxiv:1607.00148
    @Rule(
        Problem(modality='text', task='anomaly'),
        NOT(Architecture())
    )
    def af12_text_anomaly(self):
        self.declare(Architecture(family='lstm_autoencoder', is_pretrained=False))

    # ── Tabular ──────────────────────────────────────────────────────────

    # Rule AF13
    # Source: Grinsztajn et al. 2022 - arxiv:2207.08815
    # Tree models beat deep learning on small tabular data
    @Rule(
        Problem(modality='tabular',
                dataset_size=L('tiny') | L('small'),
                task=L('classify') | L('regress'),
                interpretability=False),
        NOT(Architecture())
    )
    def af13_tabular_small(self):
        self.declare(Architecture(family='xgboost', is_pretrained=False))
        self.declare(Blueprint(
            notes='Tree models outperform deep learning on small tabular data. '
                  '(Grinsztajn et al. 2022)'
        ))

    # Rule AF14
    # Source: Goodfellow et al. 2016 - deeplearningbook.org
    @Rule(
        Problem(modality='tabular',
                dataset_size=L('medium') | L('large'),
                task=L('classify') | L('regress'),
                interpretability=False),
        NOT(Architecture())
    )
    def af14_tabular_large(self):
        self.declare(Architecture(family='mlp', is_pretrained=False))

    # Rule AF15
    # Source: Rudin 2019 - arxiv:1811.10154
    # Interpretability required — no black box models
    @Rule(
        Problem(modality='tabular', task=L('classify') | L('regress'), interpretability=True),
        NOT(Architecture()),
        salience=20
    )
    def af15_tabular_interpretable(self):
        self.declare(Architecture(family='logreg', is_pretrained=False))

    # Rule AF16
    # Source: Xu et al. 2019 - arxiv:1907.00503
    @Rule(
        Problem(modality='tabular', task='generate'),
        NOT(Architecture())
    )
    def af16_tabular_generate(self):
        self.declare(Architecture(family='gan', is_pretrained=False))
        self.declare(Blueprint(
            notes='Use CTGAN for tabular data generation '
                  '(Xu et al. 2019)'
        ))

    # ── Sequential / Time Series ─────────────────────────────────────────

    # Rule AF17
    # Source: Hochreiter & Schmidhuber 1997
    @Rule(
        Problem(modality='timeseries', task='classify',
                input_shape=L('small') | L('medium')),
        NOT(Architecture())
    )
    def af17_timeseries_classify_short(self):
        self.declare(Architecture(family='lstm', is_pretrained=False))

    # Rule AF18
    # Source: Zhou et al. 2021 - arxiv:2012.07436
    @Rule(
        Problem(modality='timeseries', task='classify', input_shape='large'),
        NOT(Architecture())
    )
    def af18_timeseries_classify_long(self):
        self.declare(Architecture(family='transformer_ts', is_pretrained=False))

    # Rule AF19
    # Source: Hochreiter & Schmidhuber 1997
    @Rule(
        Problem(modality='timeseries', task='regress',
                input_shape=L('small') | L('medium')),
        NOT(Architecture())
    )
    def af19_timeseries_regress_short(self):
        self.declare(Architecture(family='lstm', is_pretrained=False))

    # Rule AF20
    # Source: Zhou et al. 2021 - arxiv:2012.07436
    @Rule(
        Problem(modality='timeseries', task='regress', input_shape='large'),
        NOT(Architecture())
    )
    def af20_timeseries_regress_long(self):
        self.declare(Architecture(family='transformer_ts', is_pretrained=False))

    # Rule AF21
    # Source: Malhotra et al. 2016 - arxiv:1607.00148
    @Rule(
        Problem(modality='timeseries', task='anomaly'),
        NOT(Architecture())
    )
    def af21_timeseries_anomaly(self):
        self.declare(Architecture(family='lstm_autoencoder', is_pretrained=False))

    # ── Audio ────────────────────────────────────────────────────────────

    # Rule AF22
    # Source: Baevski et al. 2020 - arxiv:2006.11477
    @Rule(
        Problem(modality='audio', task='classify', is_pretrained=True),
        NOT(Architecture())
    )
    def af22_audio_classify_pretrained(self):
        self.declare(Architecture(family='wav2vec', is_pretrained=True))

    # Rule AF23
    # Source: LeCun et al. 1995
    @Rule(
        Problem(modality='audio', task='classify', is_pretrained=False),
        NOT(Architecture())
    )
    def af23_audio_classify_scratch(self):
        self.declare(Architecture(family='cnn_1d', is_pretrained=False))

    # Rule AF24
    # Source: van den Oord et al. 2016 - arxiv:1609.03499
    @Rule(
        Problem(modality='audio', task='generate'),
        NOT(Architecture())
    )
    def af24_audio_generate(self):
        self.declare(Architecture(family='gan', is_pretrained=False))
        self.declare(Blueprint(
            notes='Consider WaveNet or diffusion-based audio models '
                  '(van den Oord et al. 2016)'
        ))

    # ── Special tasks ────────────────────────────────────────────────────

    # Rule AF25 — anomaly fallback
    # Source: Bank et al. 2023 - arxiv:2003.05991
    # Catches any anomaly task whose modality wasn't covered above
    @Rule(
        Problem(task='anomaly'),
        NOT(Architecture()),
        salience=-5
    )
    def af25_anomaly_fallback(self):
        self.declare(Architecture(family='autoencoder', is_pretrained=False))

    # ── Override rules ───────────────────────────────────────────────────

    # Rule AF26 — CPU override
    # Source: Howard et al. 2017 - arxiv:1704.04861
    # Transformers are not CPU-viable
    @Rule(
        Problem(compute='cpu'),
        Architecture(family=L('bert') | L('gpt') | L('transformer_ts')),
        salience=40
    )
    def af26_cpu_override(self):
        # Retract the heavy architecture and replace with lightweight
        for fact in self.facts:
            if isinstance(fact, Architecture):
                self.retract(fact)
        self.declare(Architecture(family='mlp', is_pretrained=False))
        self.declare(Blueprint(
            notes='WARNING: CPU-only detected. Heavy transformer replaced with MLP. '
                  '(Howard et al. 2017)'
        ))

    # Rule AF27 — real-time latency warning
    # Source: Sanh et al. 2019 - arxiv:1910.01108
    @Rule(
        Problem(realtime=True),
        Architecture(family=L('bert') | L('gpt')),
        salience=35
    )
    def af27_realtime_distill_note(self):
        self.declare(Blueprint(
            notes='Consider DistilBERT or a quantized model for lower latency '
                  '(Sanh et al. 2019)'
        ))

    # ── Validation rules ─────────────────────────────────────────────────

    # Rule AF28
    # Source: Logic constraint
    @Rule(
        Problem(task='regress', output_type=L('binary') | L('multiclass')),
        salience=50
    )
    def af28_regress_output_mismatch(self):
        self.declare(Blueprint(
            notes='WARNING: task=regress contradicts output_type. '
                  'Please review your answers.'
        ))

    # Rule AF29
    # Source: Logic constraint
    @Rule(
        Problem(task='classify',
                output_type=L('continuous_single') | L('continuous_multi')),
        salience=50
    )
    def af29_classify_output_mismatch(self):
        self.declare(Blueprint(
            notes='WARNING: task=classify contradicts output_type. '
                  'Please review your answers.'
        ))

    # Rule AF30
    # Source: Logic constraint
    @Rule(
        Problem(task='generate'),
        salience=40
    )
    def af30_generate_output_note(self):
        self.declare(Blueprint(
            notes='NOTE: output_type is not applicable for generation tasks.'
        ))

    # ── Fallback ─────────────────────────────────────────────────────────

    # Rule AF31
    # Source: Logic constraint
    # If nothing else matched, don't leave the user with a blank report
    @Rule(
        NOT(Architecture()),
        salience=-10
    )
    def af31_unknown_fallback(self):
        self.declare(Architecture(family='unknown', is_pretrained=False))
        self.declare(Blueprint(
            notes='WARNING: Could not determine an architecture from your '
                  'answers. Please review your inputs.'
        ))