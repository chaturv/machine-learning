import csv
from random import seed
from random import randrange
import numpy as np

import dtree

PATH = '../../dataset/data_banknote_authentication.txt'  # data file
N_FOLDS = 5  # number of random test/train splits in the ratio 80:20
MAX_DEPTH = 7  # Max depth of the decision tree
MIN_SIZE = 10  # Minimum number of data rows in leaf node


def load_csv(path):
    f = open(path, 'rb')
    lines = csv.reader(f)
    dataset = []
    for line in lines:
        dataset.append([float(col.strip()) for col in line])

    return dataset


def cross_validation_split(dataset, n_folds):
    cv_splits = []
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        cv_split = []
        while len(cv_split) < fold_size:
            index = randrange(len(dataset_copy))
            cv_split.append(dataset_copy.pop(index))
        cv_splits.append(cv_split)

    return cv_splits


def accuracy_metric(actual, predictions):
    correct = 0
    for i in xrange(len(actual)):
        if actual[i] == predictions[i]:
            correct += 1

    return 100 * correct / float(len(actual))


def evaluate_algorithm(cv_splits, *args):
    scores = []  # accuracy metric of all cv split combinations
    for cv_split in cv_splits:
        train_set = list(cv_splits)  # new list with a all splits
        train_set.remove(cv_split)  # remove the current split. this split would be used a test split
        train_set = sum(train_set, [])  # flatten the rest

        test_set = []
        for row in cv_split:
            row_copy = list(row)
            row_copy[-1] = None  # set the prediction to None
            test_set.append(row_copy)

        tree = dtree.build_tree(train_set, *args)
        predictions = []
        for row in test_set:
            prediction = dtree.predict(tree, row)
            predictions.append(prediction)

        actual = [row[-1] for row in cv_split]
        accuracy = accuracy_metric(actual, predictions)
        scores.append(accuracy)

    return scores


def run():
    seed(31)
    dataset = load_csv(PATH)
    cv_splits = cross_validation_split(dataset, N_FOLDS)
    scores = evaluate_algorithm(cv_splits, MAX_DEPTH, MIN_SIZE)

    print 'Scores : {scores}'.format(scores=scores)
    print 'Average score : {avg_score}'.format(avg_score=np.mean(scores))


if __name__ == '__main__':
    run()
