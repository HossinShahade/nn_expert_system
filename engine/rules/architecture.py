from experta import Rule, NOT, OR, L, salience
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

    # ── Text ─────────────────────────────────────────────────────────────

    # Rule AF5
    # Source: Devlin et al. 2018 - arxiv:1810.04805
    @Rule(
        Problem(modality='text', task='classify', is_pretrained=True),
        NOT(Architecture())
    )
    def af5_text_classify_pretrained(self):
        self.declare(Architecture(family='bert', is_pretrained=True))

    # Rule AF6
    # Source: Graves et al. 2013 - arxiv:1303.5778
    @Rule(
        Problem(modality='text', task='classify', is_pretrained=False),
        NOT(Architecture())
    )
    def af6_text_classify_no_pretrained(self):
        self.declare(Architecture(family='lstm', is_pretrained=False))

    # Rule AF7
    # Source: Raffel et al. 2019 - arxiv:1910.10683
    @Rule(
        Problem(modality='text', task='generate'),
        NOT(Architecture())
    )
    def af7_text_generate(self):
        self.declare(Architecture(family='gpt', is_pretrained=True))

    # ── Tabular ──────────────────────────────────────────────────────────

    # Rule AF8
    # Source: Grinsztajn et al. 2022 - arxiv:2207.08815
    # Tree models beat deep learning on small tabular data
    @Rule(
        Problem(modality='tabular',
                dataset_size=L('tiny') | L('small')),
        NOT(Architecture())
    )
    def af8_tabular_small(self):
        self.declare(Architecture(family='xgboost', is_pretrained=False))
        self.declare(Blueprint(
            notes='Tree models outperform deep learning on small tabular data. '
                  '(Grinsztajn et al. 2022)'
        ))

    # Rule AF9
    # Source: Goodfellow et al. 2016 - deeplearningbook.org
    @Rule(
        Problem(modality='tabular',
                dataset_size=L('medium') | L('large'),
                interpretability=False),
        NOT(Architecture())
    )
    def af9_tabular_large(self):
        self.declare(Architecture(family='mlp', is_pretrained=False))

    # Rule AF10
    # Source: Rudin 2019 - arxiv:1811.10154
    # Interpretability required — no black box models
    @Rule(
        Problem(modality='tabular', interpretability=True),
        NOT(Architecture())
    )
    def af10_tabular_interpretable(self):
        self.declare(Architecture(family='logreg', is_pretrained=False))

    # ── Sequential / Time Series ─────────────────────────────────────────

    # Rule AF11
    # Source: Hochreiter & Schmidhuber 1997
    @Rule(
        Problem(sequential=True, input_shape=L('small') | L('medium')),
        NOT(Architecture())
    )
    def af11_sequential_short(self):
        self.declare(Architecture(family='lstm', is_pretrained=False))

    # Rule AF12
    # Source: Zhou et al. 2021 - arxiv:2012.07436
    @Rule(
        Problem(sequential=True, input_shape='large'),
        NOT(Architecture())
    )
    def af12_sequential_long(self):
        self.declare(Architecture(family='transformer_ts', is_pretrained=False))

    # ── Special tasks ────────────────────────────────────────────────────

    # Rule AF13
    # Source: Bank et al. 2023 - arxiv:2003.05991
    @Rule(
        Problem(task='anomaly'),
        NOT(Architecture())
    )
    def af13_anomaly(self):
        self.declare(Architecture(family='autoencoder', is_pretrained=False))

    # Rule AF14
    # Source: Baevski et al. 2020 - arxiv:2006.11477
    @Rule(
        Problem(modality='audio', is_pretrained=True),
        NOT(Architecture())
    )
    def af14_audio_pretrained(self):
        self.declare(Architecture(family='wav2vec', is_pretrained=True))

    # Rule AF15
    # Source: LeCun et al. 1995
    @Rule(
        Problem(modality='audio', is_pretrained=False),
        NOT(Architecture())
    )
    def af15_audio_scratch(self):
        self.declare(Architecture(family='cnn_1d', is_pretrained=False))

    # Rule AF16 — CPU override
    # Source: Howard et al. 2017 - arxiv:1704.04861
    # Transformers are not CPU-viable
    @Rule(
        Problem(compute='cpu'),
        Architecture(family=L('bert') | L('gpt') | L('transformer_ts'))
    )
    def af16_cpu_override(self):
        # Retract the heavy architecture and replace with lightweight
        for fact in self.facts:
            if isinstance(fact, Architecture):
                self.retract(fact)
        self.declare(Architecture(family='mlp', is_pretrained=False))
        self.declare(Blueprint(
            notes='WARNING: CPU-only detected. Heavy transformer replaced with MLP. '
                  '(Howard et al. 2017)'
        ))