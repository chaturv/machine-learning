from sklearn import tree
import graphviz
from random import seed
from random import randrange

import banknotes

seed(19)
PATH = '../../dataset/data_banknote_authentication.txt'  # data file
N_FOLDS = 5  # number of random test/train splits in the ratio 80:20


def run():
    dataset = banknotes.load_csv(PATH)
    cv_splits = banknotes.cross_validation_split(dataset, N_FOLDS)
    print 'Length of cv_splits = [{l}]'.format(l=len(cv_splits))

    # create the test set
    idx_test_split = randrange(len(cv_splits))
    print 'Choosing index [{idx}] for test set'.format(idx=idx_test_split)
    test_set = cv_splits[idx_test_split]

    # create the training set
    train_set = list(cv_splits)  # new list with a all splits
    train_set.remove(test_set)  # remove the test split
    train_set = sum(train_set, [])  # flatten the rest

    x_train, y_train = split_data_target(train_set)
    x_test, y_test_actual = split_data_target(test_set)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)

    y_test_predicted = clf.predict(x_test)
    accuracy = banknotes.accuracy_metric(y_test_actual, y_test_predicted)
    print 'Accuracy = {accuracy}'.format(accuracy=accuracy)

    dot_data = tree.export_graphviz(clf,
                                    out_file='../../dataset/out/banknotes.dot',
                                    feature_names=['variance', 'skewness', 'kurtosis', 'entropy'],
                                    class_names=['authentic', 'fake'],
                                    filled=True, rounded=True)


def split_data_target(dataset):
    X = []
    Y = []
    for row in dataset:
        X.append(row[0:len(row) - 1])
        Y.append(row[-1])

    return X, Y


if __name__ == '__main__':
    run()
