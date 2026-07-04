# tests/test_rules_architecture.py

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experta import KnowledgeEngine, DefFacts, Fact
from engine.facts import Problem, Architecture, Blueprint
from engine.engine import run_engine
from engine.rules.architecture import ArchitectureRules
from report.formatter import format_report


class EngineUnderTest(ArchitectureRules, KnowledgeEngine):
    @DefFacts()
    def startup(self):
        yield Fact(started=True)


def test_image_classify_pretrained():
    engine = EngineUnderTest()
    engine.reset()
    engine.declare(Problem(
        modality='image',
        task='classify',
        is_pretrained=True,
        dataset_size='small'
    ))
    engine.run()

    architectures = [f for f in engine.facts.values() if isinstance(f, Architecture)]
    assert len(architectures) == 1
    assert architectures[0]['family'] == 'cnn_pretrained'
    print("PASS: image + classify + pretrained -> cnn_pretrained")


def test_tabular_small_forces_xgboost():
    engine = EngineUnderTest()
    engine.reset()
    engine.declare(Problem(
        modality='tabular',
        task='classify',
        dataset_size='small',
        interpretability=False
    ))
    engine.run()

    architectures = [f for f in engine.facts.values() if isinstance(f, Architecture)]
    assert architectures[0]['family'] == 'xgboost'
    print("PASS: tabular + small -> xgboost")


def test_run_engine_returns_architecture_and_blueprints():
    answers = {
        'modality': 'image',
        'task': 'classify',
        'output_type': 'binary',
        'dataset_size': 'small',
        'sequential': False,
        'spatial': True,
        'is_pretrained': True,
        'realtime': False,
        'compute': 'cpu',
        'class_imbalance': False,
        'interpretability': False,
        'input_shape': 'small',
        'outliers_expected': False,
    }

    architecture, blueprints = run_engine(answers)

    assert architecture is not None
    assert architecture['family'] == 'cnn_pretrained'
    assert len(blueprints) >= 1


def test_format_report_uses_single_value_per_field_and_includes_output_activation():
    architecture = {'family': 'cnn_pretrained'}
    blueprints = [
        Blueprint(output_layer='Linear(last_hidden -> num_classes)', output_activation='None -- BCEWithLogitsLoss applies Sigmoid internally'),
        Blueprint(output_layer='Linear(last_hidden -> 1)', output_activation='None -- linear output, no activation needed'),
    ]

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        format_report(architecture, blueprints)

    output = buffer.getvalue()
    assert 'OUTPUT LAYER' in output
    assert 'OUTPUT ACTIVATION' in output
    assert 'linear output' in output
    assert 'BCEWithLogitsLoss' not in output


if __name__ == "__main__":
    test_image_classify_pretrained()
    test_tabular_small_forces_xgboost()
    test_run_engine_returns_architecture_and_blueprints()