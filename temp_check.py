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
    count += 1
    if count % 1000 == 0:
        print(f'Progress: tested {count}/{total_combinations} combinations so far')

print('Finished.')
print('tested', count, 'combinations')
print('unknown_count', len(unknown))
for item in unknown:
    print(item)
