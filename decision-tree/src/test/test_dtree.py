import unittest
import dtree.dtree as dtree


class TestDTree(unittest.TestCase):
    def test_gini_index_non_zero(self):
        # four rows with one having a different class
        group_1 = [[1, 2, 0], [1, 3, 0], [1, 4, 0], [1, 1.5, 1]]
        # one row
        group_2 = [[2, 1, 1]]

        groups = [group_1, group_2]
        classes = [0, 1]

        expected = 0.3
        self.assertEqual(dtree.gini_index(groups, classes), expected)

    def test_gini_index_zero(self):
        # same class
        group_1 = [[1, 2, 0], [1, 3, 0], [1, 4, 0]]
        # same class
        group_2 = [[2, 1, 1], [2, 2.5, 1]]

        groups = [group_1, group_2]
        classes = [0, 1]

        expected = 0.0
        self.assertEqual(dtree.gini_index(groups, classes), expected)

    def test_get_split(self):
        dataset = [[2.771244718, 1.784783929, 0],
                   [1.728571309, 1.169761413, 0],
                   [3.678319846, 2.81281357, 0],
                   [3.961043357, 2.61995032, 0],
                   [2.999208922, 2.209014212, 0],
                   [7.497545867, 3.162953546, 1],
                   [9.00220326, 3.339047188, 1],
                   [7.444542326, 0.476683375, 1],
                   [10.12493903, 3.234550982, 1],
                   [6.642287351, 3.319983761, 1]]

        split = dtree.get_split(dataset)

        self.assertEqual(1, split['index'] + 1)
        self.assertEqual(6.642287351, split['value'])

    def test_to_terminal(self):
        # three rows with class 2 and one each with 1 and 0
        group_1 = [[1, 2, 2], [1, 3, 2], [1, 4, 2], [1, 5, 1], [1, 5, 0]]
        outcome = dtree.to_terminal(group_1)
        self.assertEqual(outcome, 2)

    def test_build_tree(self):
        dataset = [[2.771244718, 1.784783929, 0],
                   [1.728571309, 1.169761413, 0],
                   [3.678319846, 2.81281357, 0],
                   [3.961043357, 2.61995032, 0],
                   [2.999208922, 2.209014212, 0],
                   [7.497545867, 3.162953546, 1],
                   [9.00220326, 3.339047188, 1],
                   [7.444542326, 0.476683375, 1],
                   [10.12493903, 3.234550982, 1],
                   [6.642287351, 3.319983761, 1]]
        tree = dtree.build_tree(dataset, 2, 1)
        self.assertIsNotNone(tree)

    def test_predict(self):
        dataset = [[2.771244718, 1.784783929, 0],
                   [1.728571309, 1.169761413, 0],
                   [3.678319846, 2.81281357, 0],
                   [3.961043357, 2.61995032, 0],
                   [2.999208922, 2.209014212, 0],
                   [7.497545867, 3.162953546, 1],
                   [9.00220326, 3.339047188, 1],
                   [7.444542326, 0.476683375, 1],
                   [10.12493903, 3.234550982, 1],
                   [6.642287351, 3.319983761, 1]]

        simple_node = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
        for row in dataset:
            prediction = dtree.predict(simple_node, row)
            print('Expected=%d, Got=%d' % (row[-1], prediction))
            self.assertEqual(row[-1], prediction)
