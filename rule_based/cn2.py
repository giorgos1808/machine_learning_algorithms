# HOMEWORK 3 - RULE-BASED LEARNING
# CN2 ALGORITHM TEMPLATE
# Tsakiris Giorgos

from Orange.classification import CN2Learner, CN2UnorderedLearner
import Orange.datasets
from Orange.classification.rules import EntropyEvaluator, LaplaceAccuracyEvaluator

wineData = Orange.data.Table('wine')

Ordered_learner = CN2Learner()
Unordered_learner = CN2UnorderedLearner()


def rule_learner(learner, heuristic='Entropy'):
    f1 = -10
    accuracy = 0
    recall = 0
    precision = 0
    best = [0, 0, 0]

    if heuristic == 'Entropy':
        evaluator = EntropyEvaluator()
    elif heuristic == 'Laplace':
        evaluator = LaplaceAccuracyEvaluator()

    for Beam_Width in range(3, 6):
        for Min_Covered_Examples in range(7, 9):
            for Max_Rule_Length in range(2, 4):
                learner.rule_finder.quality_evaluator = evaluator
                learner.rule_finder.search_algorithm.beam_width = Beam_Width
                learner.rule_finder.general_validator.min_covered_examples = Min_Covered_Examples
                learner.rule_finder.general_validator.max_rule_length = Max_Rule_Length

                results = Orange.evaluation.testing.CrossValidation(wineData, [learner], k=10)

                if f1 < Orange.evaluation.scoring.F1(results, average='micro'):
                    f1 = Orange.evaluation.scoring.F1(results, average='micro')
                    accuracy = Orange.evaluation.scoring.CA(results)
                    recall = Orange.evaluation.scoring.Recall(results, average='micro')
                    precision = Orange.evaluation.scoring.Precision(results, average='micro')

                    best = [Beam_Width, Min_Covered_Examples, Max_Rule_Length]
                    model = learner

    print('Accuracy: %f' % accuracy)
    print('Recall: %f' % recall)
    print('Precision: %f' % precision)
    print('F-measure: %f' % f1)
    print('Beam Width: %d , Min Covered Examples: %d , Max Rule Length %d' % (best[0], best[1], best[2]))

    classifier = model(wineData)
    print('\nRules')
    for rule in classifier.rule_list:
        print(rule)


rule_learner(Ordered_learner, 'Entropy')
rule_learner(Unordered_learner, 'Laplace')
rule_learner(Ordered_learner, 'Laplace')
