from experta import KnowledgeEngine, DefFacts, Fact

from engine.facts import Problem, Architecture, Blueprint
from engine.rules.architecture  import ArchitectureRules
from engine.rules.depth         import DepthRules
from engine.rules.width         import WidthRules
from engine.rules.input_layer   import InputLayerRules
from engine.rules.output_layer  import OutputLayerRules
from engine.rules.activation    import ActivationRules
from engine.rules.normalization import NormalizationRules
from engine.rules.dropout       import DropoutRules
from engine.rules.loss          import LossRules
from engine.rules.optimizer     import OptimizerRules
from engine.rules.init          import InitRules


class NNExpertSystem(
    ArchitectureRules,
    DepthRules,
    WidthRules,
    InputLayerRules,
    OutputLayerRules,
    ActivationRules,
    NormalizationRules,
    DropoutRules,
    LossRules,
    OptimizerRules,
    InitRules,
    KnowledgeEngine
):
    @DefFacts()
    def startup(self):
        yield Fact(started=True)


def run_engine(answers: dict):
    """
    Takes the answers dict from the CLI,
    declares them as facts, runs the engine,
    and returns all Blueprint facts.
    """
    engine = NNExpertSystem()
    engine.reset()

    # Declare each answer as a field on a Problem fact
    engine.declare(Problem(**answers))

    # Run inference — fires all matching rules
    engine.run()

    # Collect all Blueprint facts from the fact base
    blueprints = []
    for fact in engine.facts.values():
        if isinstance(fact, Blueprint):
            blueprints.append(fact)

    # Collect the Architecture fact
    architecture = None
    for fact in engine.facts.values():
        if isinstance(fact, Architecture):
            architecture = fact
            break

    return architecture, blueprints