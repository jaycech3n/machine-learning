# ILAS Exercise set 2 - Decision tree learning
# Task 1.

import graphviz as gv
import pandas as pd
import math

class DecisionTree:
    '''Decision tree with continuous attribute values and binary target attribute.'''


    class ContinuousBinaryNode:
        '''Node class for a continuous attribute binary decision tree.
        Nodes are stored in a list `self.nodes`, and connectivity is handled by the individual nodes.

        Class variables:
        self.data - DataFrame containing the instances corresponding to this node in the decision tree.
        self.test - Attribute name string to test at the root. `None` if we are a leaf.
        self.splitvalue - Attribute value to split by. `None` if we are a leaf.
        self.left, self.right - Integers giving the indices of the left and right child nodes respectively. `None` if we are a leaf.
        self.targetclass - Target class value. `None` if we are an internal (i.e. a test attribute) node.'''

        def __init__(self, data, test=None, splitvalue=None, left=None, right=None, targetclass=None):
            self.data = data
            self.test = test
            self.splitvalue = splitvalue
            self.left = left
            self.right = left
            self.targetclass = targetclass

        def __repr__(self):
            if not self.test and not self.splitvalue:
                if self.targetclass is None:
                    return "Uninitialized node"
                else:
                    return "Leaf: Target class = " + str(self.targetclass)
            else:
                return "Test node: " + self.test + " < " + str(self.splitvalue) + ", left: " + str(self.left) + ", right: " + str(self.right)


    def __init__(self, data, targetattribute=None, posvalue=1.0, negvalue=0.0):
        '''Parameters:
        data - pandas DataFrame of the data to be classified.
        target - String of the column name of the target attribute.'''

        self.nodes = [self.ContinuousBinaryNode(data)]
        self.targetattribute = targetattribute if targetattribute else data.columns.values[-1]
        self.posvalue = posvalue
        self.negvalue = negvalue


    def __repr__(self):
        return "[" + ", ".join(str(node) for node in self.nodes) + "]"


    def build(self):
        '''Construct the decision tree, breadth-first.'''

        def candidate_splitvalues(data, attribute):
            sorted_data = data.sort(attribute)
            labels = sorted_data[self.targetattribute].values
            indices = [i for i in range(len(labels)) if i < len(labels)-1 and labels[i] != labels[i+1]]
            attributevalues = sorted_data[attribute].values
            return [(attributevalues[i] + attributevalues[i+1]) / 2.0 for i in indices if attributevalues[i] != attributevalues[i+1] or i > 0]

            # Remark: In the case that two instances with the same value v for a particular attribute A are
            # classified differently, the split value will be v, and the cases will be A < v and A >= v.
            # If such a situation occurs for two instances in positions 0 and 1 of the sorted data then we
            # would get an empty collection for the case A < v, so we do not want to return v in this case.
            # The guard in the return statement above ensures that every v in the returned list does indeed
            # split the data.

        def entropy(data):
            log2 = lambda x : 0 if x == 0 else math.log(x, 2)
            N = len(data)
            p0 = len(data.loc[data[self.targetattribute] == self.negvalue]) / N
            p1 = 1 - p0
            return -p0 * log2(p0) - p1 * log2(p1)

        def gain(data, attribute, split_value):
            N = len(data)
            data_lt = data.loc[data[attribute] < split_value]
            data_geq = data.loc[data[attribute] >= split_value]
            p_lt = len(data_lt) / N
            p_geq = 1 - p_lt
            return entropy(data) - p_lt * entropy(data_lt) - p_geq * entropy(data_geq)

        for node in self.nodes:
            data = node.data
            positives = data.loc[data[self.targetattribute] == self.posvalue]

            # Case 1: all instances have the same target attribute value. Then classify.
            if len(positives) == 0 or len(positives) == len(data):
                node.targetclass = self.negvalue if len(positives) == 0 else self.posvalue
            else:
                attributes = data.columns.values[:-1]  # Get attributes to test. Last column is the target attribute.
                candidate_attributes = {}  # Candidate attributes to split on, together with their candidate split values.
                for attribute in attributes:
                    candidatesplits = candidate_splitvalues(data, attribute)
                    if candidatesplits:
                        candidate_attributes[attribute] = candidatesplits
                # Case 2: no test splits the data. Then assign the majority label to this node.
                if not candidate_attributes:
                    node.targetclass = self.posvalue if 2 * len(positives) >= len(data) else self.negvalue
                # Case 3: this node is a test node. Find the attribute and split value with the highest gain, and set the parameters.
                else:
                    best_gain = 0
                    best_attribute = ""
                    best_split_value = 0
                    for attribute in candidate_attributes:
                        for split_value in candidate_attributes[attribute]:
                            g = gain(data, attribute, split_value)
                            if g > best_gain:
                                best_gain = g
                                best_attribute = attribute
                                best_split_value = split_value

                    # Set node parameters.
                    node.test = best_attribute
                    node.splitvalue = best_split_value
                    data_lt = data.loc[data[best_attribute] < best_split_value]  #.drop(best_attribute, 1)  # Also remove the attribute from the dataset, if we want.
                    data_geq = data.loc[data[best_attribute] >= best_split_value]  #.drop(best_attribute, 1)

                    # Set child nodes.
                    node.left = len(self.nodes)
                    self.nodes.append(self.ContinuousBinaryNode(data_lt))
                    node.right = len(self.nodes)
                    self.nodes.append(self.ContinuousBinaryNode(data_geq))


    def render(self, outputname=None):
        '''Render decision tree as a pdf.'''
        self.dot = gv.Digraph(outputname)
        for id in range(len(self.nodes)):
            self.dot.node(str(id))
        for id, node in enumerate(self.nodes):
            if node.test:  # Test node
                self.dot.node(str(id), label=node.test, shape="box")
                self.dot.edge(str(id), str(node.left), "< " + str(node.splitvalue))
                self.dot.edge(str(id), str(node.right), ">= " + str(node.splitvalue))
            else:  # Leaf
                self.dot.node(str(id), label=str(int(node.targetclass)), shape="circle")
        self.dot.render(view = True)


    def classify(self, instance):
        '''Classifies an instance using the trained decision tree.
        instance is a pandas Series.'''

        def _classify(self, instance, nodeindex):
            node = self.nodes[nodeindex]
            if node.targetclass != None:
                return node.targetclass
            else:
                attr = node.test
                splitvalue = node.splitvalue
                if instance[attr] < splitvalue:
                    return _classify(self, instance, node.left)
                else:
                    return _classify(self, instance, node.right)

        return _classify(self, instance, 0)

# Task 2.

trainingfile = r'gene_expression_training.csv'
testfile = r'gene_expression_test.csv'

data_train = pd.read_csv(trainingfile)
data_test = pd.read_csv(testfile)

# Build the classifier, output dot file to "Digraph.gv" and render pdf.
T = DecisionTree(data_train)
T.build()
T.render()

# Task 3.
# Test on test data set.

predictions = [T.classify(instance) for id, instance in data_test.iterrows()]
classes = list(data_test['class_label'])

correct_predictions = sum(int(predictions[i] == classes[i]) for i in range(len(classes)))
accuracy = correct_predictions / len(classes)

print("Test accuracy: " + str(accuracy))
