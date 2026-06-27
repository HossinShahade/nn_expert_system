from engine.facts import Problem, Architecture, Blueprint
p = Problem(modality='image', task='classify')
print(p) 
from cli.questions import ask_questions
answers = ask_questions()
print(answers)