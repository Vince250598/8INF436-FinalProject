import utilDecisionTreeClassification


def applySupervisedModel(X_train, y_train, X_test):
    decision_tree_parameters = utilDecisionTreeClassification.DecisionTreeParameters()
    decision_tree_parameters.criterion = 'entropy'
    decision_tree_parameters.splitter = 'best'
    decision_tree_parameters.max_depth = None
    decision_tree_parameters.min_samples_split = 2
    decision_tree_parameters.min_samples_leaf = 1
    decision_tree_parameters.max_feature = 'sqrt'
    decision_tree_parameters.max_leaf_nodes = None

    print("The decision tree algorithm is executing. Please wait ...")

    # Create and train the model of the decision tree
    decision_tree_classifier, training_running_time = \
        utilDecisionTreeClassification.train_decision_tree_classifier(X_train, y_train, decision_tree_parameters)

    print("The training process of the model of the decision tree took : %.8f second" % training_running_time)

    X_test['cluster'] = decision_tree_classifier.predict(X_test)

    return X_test