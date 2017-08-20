# Make a prediction
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# Return the most common class value in this terminal group
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    # apply the list.count function with outcomes (0, 1, ..) as input and return max
    return max(set(outcomes), key=outcomes.count)


# Select the best split point for a dataset
def get_split(dataset):
    # get distinct class values
    class_values = list(set([row[-1] for row in dataset]))
    # the best of everything
    b_index, b_value, b_score, b_groups = None, None, None, None
    for index in xrange(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            print('X%d < %.3f Gini=%.3f' % ((index + 1), row[index], gini))
            if b_score is None or gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups

    splits = {'index': b_index, 'value': b_value, 'groups': b_groups}
    print('** Best split: [X%d < %.3f]' % ((splits['index'] + 1), splits['value']))
    return splits


# Split the dataset into left or right rows given an attribute index and split value
def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        left.append(row) if row[index] < value else right.append(row)

    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # get the total count of data set
    n_instances = float(sum([len(group) for group in groups]))

    gini = 0.0
    for group in groups:
        # size of this group
        size_grp = float(len(group))
        if size_grp == 0:
            continue

        # for each class, get the proportion
        score = 0.0
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / size_grp
            score += proportion * proportion
        # calculate gini for this group and add to previous
        gini += (1.0 - score) * size_grp / n_instances

    return gini
