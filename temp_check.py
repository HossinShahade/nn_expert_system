from itertools import product
from cli.questions import QUESTIONS
from engine.engine import run_engine

choices = [[opt[0] for opt in q['options']] for q in QUESTIONS]
fields = [q['field'] for q in QUESTIONS]
base = {'outliers_expected': False}

total_combinations = 1
for choice_list in choices:
    total_combinations *= len(choice_list)

unknown = []
count = 0
print('Starting coverage check...')
print(f'Expecting {total_combinations} total combinations')
for combo in product(*choices):
    answers = dict(zip(fields, combo))
    answers.update(base)
    arch, _ = run_engine(answers)
    if arch is None or arch['family'] == 'unknown':
        unknown.append((answers['modality'], answers['task'], answers['output_type'], answers['dataset_size'], answers['sequential'], answers['spatial'], answers['is_pretrained'], answers['realtime'], answers['compute'], answers['class_imbalance'], answers['interpretability'], answers['input_shape']))
        print('Found unknown fallback:', answers)
    # New: detect conflicting blueprint field values (more than one distinct value per combo)
    arch, blueprints = run_engine(answers)
    # fields to check for collisions
    check_fields = ['input_layer', 'hidden_layers', 'width', 'output_layer', 'output_activation', 'activation', 'normalization', 'dropout_rate', 'dropout_placement', 'loss', 'optimizer', 'lr', 'lr_schedule', 'init']
    collisions = {}
    for f in check_fields:
        vals = set()
        for bp in blueprints:
            v = bp.get(f)
            if v is not None:
                vals.add(v)
        if len(vals) > 1:
            collisions[f] = vals
    if collisions:
        print('Conflict detected for combo:', answers)
        for k, vset in collisions.items():
            print(f'  Field "{k}" has {len(vset)} distinct values:')
            for v in list(vset)[:5]:
                print(f'    - {v}')
        # optional: record or break; here we just record one example
        unknown.append(('conflict', answers, collisions))
    count += 1
    if count % 1000 == 0:
        print(f'Progress: tested {count}/{total_combinations} combinations so far')

print('Finished.')
print('tested', count, 'combinations')
print('unknown_count', len(unknown))
for item in unknown:
    print(item)
