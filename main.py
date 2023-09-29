import numpy
# import matplotlib
# matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import linear_model, tree, metrics, datasets, set_config
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, LeaveOneOut, LeavePOut, ShuffleSplit
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier


def standart_deviation():
    speed = [40, 42, 59, 70, 113, 37, 54, 88, 32, 47, 50]
    std_speed = numpy.std(speed)
    return std_speed


def variance():
    speed = [40, 42, 59, 70, 113, 37, 54, 88, 32, 47, 50]
    var_speed = numpy.var(speed)
    return var_speed


def percentile():
    """ What is the age that 90% of the people are younger than ... ? """

    age = [13, 20, 34, 78, 16, 29, 31, 80, 65, 47]
    perc = numpy.percentile(age, 90)
    return perc


def data_set():
    set = numpy.random.uniform(1.0, 500000000000.0, 2500)
    plt.hist(set, 5)
    return plt.show()


def gaussian_data_distribution():
    set = numpy.random.normal(5.0, 1.0, 600000)
    plt.hist(set, 100)
    return plt.show()


def plot():
    car_age = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
    car_speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
    plt.scatter(car_age, car_speed)
    plt.xlabel('Car age')
    plt.ylabel('Car speed')
    return plt.show()


def random_data_distribution():
    x = numpy.random.normal(5.0, 1.0, 1000)
    y = numpy.random.normal(10.0, 2.0, 1000)
    plt.scatter(x, y)
    return plt.show()


def linear_regression():
    """ r - coefficient of correlation """

    x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
    y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

    slope, intercept, r, p, std_err = stats.linregress(x, y)

    def my_func(x):
        return slope * x + intercept

    mymodel = list(map(my_func, x))

    # speed = my_func(10)
    # print('speed', speed)

    plt.scatter(x, y)
    plt.plot(x, mymodel)
    print('r', r)
    return plt.show()


def polinomial_regression():
    x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
    y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]

    mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

    myline = numpy.linspace(1, 22, 100)

    # speed = mymodel(17)
    # print('speed', speed)

    plt.scatter(x, y)
    plt.plot(myline, mymodel(myline))
    print(r2_score(y, mymodel(x)))
    return plt.show()


def multiple_regression():
    df = pd.read_csv('data.csv')
    x = df[['Weight', 'Volume']]
    y = df['CO2']

    regr = linear_model.LinearRegression()
    regr.fit(x.values, y.values)

    # predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
    predictedCO2 = regr.predict([[2300, 1300]])
    # Coefficient values tell us that if the weight increase by 1kg, the CO2 emission increases by 0.00755095g,
    # and if the engine size (Volume) increases by 1 cm3, the CO2 emission increases by 0.00780526 g.
    print('Coefficient: ', regr.coef_)
    return predictedCO2


def second_example_multiple_regression():
    df = pd.read_csv('data.csv')
    x = df[['Weight', 'Volume']]
    y = df['CO2']

    regr = linear_model.LinearRegression()
    regr.fit(x.values, y.values)

    # predict the CO2 emission of a car where the weight is 3300kg, and the volume is 1300cm3:
    predictedCO2 = regr.predict([[3300, 1300]])
    return predictedCO2


def scale_values():
    scale = StandardScaler()

    df = pd.read_csv('data.csv')
    x = df[['Weight', 'Volume']]

    scaledX = scale.fit_transform(x)
    return scaledX


# ???
def second_scaled_values():
    scale = StandardScaler()

    df = pd.read_csv('data.csv')
    x = df[['Weight', 'Volume']]
    y = df['CO2']

    scaledX = scale.fit_transform(x)

    regr = linear_model.LinearRegression()
    regr.fit(scaledX, y.values)

    scaled = scale.transform([[2300, 1.3]])
    predictedCO2 = regr.predict([scaled[0]])
    return predictedCO2


def train_test():
    """ The x axis represents the number of minutes before making a purchase.
        The y axis represents the amount of money spent on the purchase."""

    numpy.random.seed(2)

    x = numpy.random.normal(3, 1, 100)
    y = numpy.random.normal(150, 40, 100)/x

    # Split Into Train/Test
    # The training set should be a random selection of 80% of the original data.
    # The testing set should be the remaining 20%.
    train_x = x[:80]
    train_y = y[:80]

    test_x = x[80:]
    test_y = y[80:]

    mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
    myline = numpy.linspace(0, 6, 100)

    plt.scatter(train_x, train_y)
    # plt.scatter(test_x, test_y)
    plt.plot(myline, mymodel(myline))

    # r2 = r2_score(train_y, mymodel(train_x))
    r2 = r2_score(test_y, mymodel(test_x))
    print('r2: ', r2)

    # How much money will a buying customer spend, if she or he stays in the shop for 5 minutes?
    print('Predicted: ', mymodel(5))
    return plt.show()


def decision_tree():
    """ To make a decision tree, all data has to be numerical. """

    df = pd.read_csv('data_tree.csv')
    # Change string values into numerical values:
    d = {'UK': 0, 'USA': 1, 'N': 2}
    df['Nationality'] = df['Nationality'].map(d)
    d = {'YES': 1, 'NO': 0}
    df['Go'] = df['Go'].map(d)

    # separate the feature columns(we try to predict from) from the target column(we try to predict).
    features = ['Age', 'Experience', 'Rank', 'Nationality']
    x = df[features]
    y = df['Go']

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(x.values, y.values)
    tree.plot_tree(dtree, feature_names=features)
    # save tree to png:
    plt.savefig('./1.png')
    # Should I go see a show starring a 40 years old American comedian,
    # with 10 years of experience, and a comedy ranking of 7?
    print(dtree.predict([[40, 10, 7, 1]]))
    print("[1] means 'GO'")
    print("[0] means 'NO'")
    return


def conf_matrix():
    actual = numpy.random.binomial(1, .9, 1000)
    predicted = numpy.random.binomial(1, .9, 1000)
    accuracy = metrics.accuracy_score(actual, predicted)
    print('accuracy: ', accuracy)
    precision = metrics.precision_score(actual, predicted)
    print('precision: ', precision)
    sensitivity_recall = metrics.recall_score(actual, predicted)
    print('sensitivity_recall: ', sensitivity_recall)
    specificity = metrics.recall_score(actual, predicted, pos_label=0)
    print('specificity: ', specificity)
    f1_score = metrics.f1_score(actual, predicted)
    print('f1_score: ', f1_score)
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    cm_display.plot()
    return plt.show()


def hierarchical_clustering():
    x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
    y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
    data = list(zip(x, y))
    linkage_data = linkage(data, method='ward', metric='euclidean')
    dendrogram(linkage_data)
    # or the same with scikit-learn library:
    # hierarchical_cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
    # labels = hierarchical_cluster.fit_predict(data)
    # plt.scatter(x, y, c=labels)
    return plt.show()


def logistic_regression_binomial():
    # X represents the size of a tumor in centimeters.
    x = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)

    # X has to be reshaped into a column from a row for the LogisticRegression() function to work.
    # y represents if the tumor is cancerous (0 for "No", 1 for "Yes").
    y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    logr = linear_model.LogisticRegression()
    logr.fit(x, y)

    log_odds = logr.coef_
    odds = numpy.exp(log_odds)
    # tells us that as the size of a tumor increases by 1mm the odds of it being a cancerous tumor increases by 4x.
    print('odds: ', odds)
    # predict if tumor is cancerous where the size is 3.46mm:
    predicted = logr.predict(numpy.array([3.46]).reshape(-1, 1))
    return predicted


def probability():
    # 3.78 0.61 The probability that a tumor with the size 3.78cm is cancerous is 61% etc.
    x = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)
    y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    logr = linear_model.LogisticRegression()
    logr.fit(x, y)

    log_odds = logr.coef_ * x + logr.intercept_
    odds = numpy.exp(log_odds)
    prob = odds / (1 + odds)
    return prob


def default_c():
    # load in the dataset.
    iris = datasets.load_iris()

    x = iris['data']
    y = iris['target']

    logit = LogisticRegression(max_iter=10000)
    print(logit.fit(x, y))
    print(logit.score(x, y))
    return


def grid_search():
    iris = datasets.load_iris()
    x = iris['data']
    y = iris['target']

    logit = LogisticRegression(max_iter=10000)

    C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    scores = []

    for choice in C:
        logit.set_params(C=choice)
        logit.fit(x, y)
        scores.append((logit.score(x, y)))
    return scores


def categorical_data_one_hot_encoding():
    cars = pd.read_csv('data.csv')
    ohe_cars = pd.get_dummies(cars['Car'], dtype=int)
    x = pd.concat([cars[['Volume', 'Weight']], ohe_cars], axis=1)
    print(ohe_cars.to_string())
    y = cars['CO2']
    regr = linear_model.LinearRegression()
    regr.fit(x.values, y.values)
    # predict the CO2 emission of a Volvo where the weight is 2300kg, and the volume is 1300cm3:
    predictedCO2 = regr.predict([[2300, 1300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
    return predictedCO2


def dummifying():
    colors = pd.DataFrame({'color': ['blue', 'red', 'green']})
    # add to .get_dummies(drop_first=True)
    dummies = pd.get_dummies(colors, dtype=int)
    dummies['color'] = colors['color']
    return dummies


def k_means():
    x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
    y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

    # Turn the data into a set of points:
    data = list(zip(x, y))

    # Before finding the good K value:

    # inertias = []
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters=i)
    #     kmeans.fit(data)
    #     inertias.append(kmeans.inertia_)

    # plt.plot(range(1, 11), inertias, marker='o')
    # plt.title('Elbow method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Inertia')

    # The elbow method shows that 2 is a good value for K, so we retrain and visualize the result:
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)
    plt.scatter(x, y, c=kmeans.labels_)
    return plt.show()


def bagging():
    data = datasets.load_wine()

    x = data.data
    y = data.target

    # split X and y into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=22)
    # instantiate a base classifier and fit it to the training data
    dtree = DecisionTreeClassifier(random_state=22)
    dtree.fit(x_train, y_train)
    # predict the class of wine the unseen test set and evaluate the model performance
    y_pred = dtree.predict(x_test)

    print("Train data accuracy:", accuracy_score(y_true=y_train, y_pred=dtree.predict(x_train)))
    print("Test data accuracy:", accuracy_score(y_true=y_test, y_pred=y_pred))
    return


def creating_bagging_classifier():
    data = datasets.load_wine()
    x = data.data
    y = data.target
    # split X and y into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=22)

    estimator_range = [2, 4, 6, 8, 10, 12, 14, 16]
    models = []
    scores = []

    for n_estimators in estimator_range:
        # Create bagging classifier
        clf = BaggingClassifier(n_estimators=n_estimators, random_state=22)
        # Fit the model
        clf.fit(x_train, y_train)
        # Append the model and score to their respective list
        models.append(clf)
        scores.append(accuracy_score(y_true=y_test, y_pred=clf.predict(x_test)))

    # Generate the plot of scores against number of estimators
    plt.figure(figsize=(9, 6))
    plt.plot(estimator_range, scores)

    # Adjust labels and font
    plt.xlabel("n_estimators", fontsize=18)
    plt.ylabel("score", fontsize=18)
    plt.tick_params(labelsize=16)
    # increase in model performance from 82.2% to 95.5% ( 10-14 estimators )
    return plt.show()


def out_of_bag_score():
    data = datasets.load_wine()
    x = data.data
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=22)
    # n_estimators value from previous function
    oob_model = BaggingClassifier(n_estimators=12, oob_score=True, random_state=22)
    oob_model.fit(x_train, y_train)
    return oob_model.oob_score_


def decision_tree_from_bagging_classifier():
    data = datasets.load_wine()
    x = data.data
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=22)

    oob_model = BaggingClassifier(n_estimators=12, oob_score=True, random_state=22)
    oob_model.fit(x_train, y_train)

    clf = BaggingClassifier(n_estimators=12, oob_score=True, random_state=22)
    clf.fit(x_train, y_train)

    plt.figure(figsize=(30, 20))
    # Add feature_names if output will be pd.DataFrame, not numpy.ndarray
    plot_tree(clf.estimators_[0])
    plt.savefig('./2.png')
    return


def cross_validation():
    x, y = datasets.load_iris(return_X_y=True)
    # k-fold cross validation
    # create and fit a model for evaluation
    clf = DecisionTreeClassifier(random_state=42)
    # evaluate the model and see how it performs on each k-fold
    k_folds = KFold(n_splits=5)
    # In cases where classes are imbalanced we need a way to account for the imbalance
    # in both the train and validation sets:
    # sk_folds = StratifiedKFold(n_splits=5)
    # Leave-One_out (Change cv=loo in scores):
    # loo = LeaveOneOut()
    # Leave-P-Out (Change cv=lpo in scores):
    # lpo = LeavePOut(p=2)
    # Shuffle Split (Change cv=ss in scores):
    # ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits = 5)
    scores = cross_val_score(clf, x, y, cv=k_folds)
    print('Cross validation scores: ', scores)
    print('Avarage cv score: ', scores.mean())
    print('Number of cv scores used in avarage: ', len(scores))
    return


def imbalanced_data():
    n = 10000
    ratio = .95
    n_0 = int((1-ratio) * n)
    n_1 = int(ratio * n)

    y = numpy.array([0] * n_0 + [1] * n_1)

    def plot_roc_curve(true_y, y_prob):
        """
        plots the roc curve based of the probabilities
        """

        fpr, tpr, thresholds = roc_curve(true_y, y_prob)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    # below are the probabilities obtained from a hypothetical model that always (!) predicts the majority class
    # probability of predicting class 1 is going to be 100%:
    y_proba = numpy.array([1] * n)
    y_pred = y_proba > .5

    plot_roc_curve(y, y_proba)
    plt.show()
    print(f'model 1 AUC score: {roc_auc_score(y, y_proba)}')
    print(f'accuracy score: {accuracy_score(y, y_pred)}')
    cf_mat = confusion_matrix(y, y_pred)

    # # below are the probabilities obtained from a hypothetical model that doesn't (!) always predict the mode
    # y_proba_2 = numpy.array(
    #     numpy.random.uniform(0, .7, n_0).tolist() +
    #     numpy.random.uniform(.3, 1, n_1).tolist()
    # )
    # y_pred_2 = y_proba_2 > .5

    # plot_roc_curve(y, y_proba_2)
    # plt.show()
    # print(f'model 2 AUC score: {roc_auc_score(y, y_proba_2)}')
    # print(f'accuracy score: {accuracy_score(y, y_pred_2)}')
    # cf_mat = confusion_matrix(y, y_pred_2)

    print('Confusion matrix')
    print(cf_mat)
    print(f'class 0 accuracy: {cf_mat[0][0] / n_0}')
    print(f'class 1 accuracy: {cf_mat[1][1] / n_1}')
    return


def two_sets_of_probabilities():
    n = 10000
    y = numpy.array([0] * n + [1] * n)

    y_prob_1 = numpy.array(
        numpy.random.uniform(.25, .5, n // 2).tolist() +
        numpy.random.uniform(.3, .7, n).tolist() +
        numpy.random.uniform(.5, .75, n // 2).tolist()
    )
    y_prob_2 = numpy.array(
        numpy.random.uniform(0, .4, n // 2).tolist() +
        numpy.random.uniform(.3, .7, n).tolist() +
        numpy.random.uniform(.6, 1, n // 2).tolist()
    )

    print(f'model 1 accuracy score: {accuracy_score(y, y_prob_1 > .5)}')
    print(f'model 2 accuracy score: {accuracy_score(y, y_prob_2 > .5)}')

    print(f'model 1 AUC score: {roc_auc_score(y, y_prob_1)}')
    print(f'model 2 AUC score: {roc_auc_score(y, y_prob_2)}')

    def plot_roc_curve(true_y, y_prob):
        """
        plots the roc curve based of the probabilities
        """
        # For model 1:
        fpr, tpr, thresholds = roc_curve(true_y, y_prob)
        plt.plot(fpr, tpr)
        # For model 2:
        # fpr, tpr, thresholds = roc_curve(y, y_prob_2)
        # plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    # Plot model 1:
    plot_roc_curve(y, y_prob_1)
    # Plot model 2:
    # plot_roc_curve(y, y_prob_2)

    plt.show()
    return


def k_nearest_neighbors():
    x = [4, 5, 10, 4, 3, 11, 14, 8, 10, 12]
    y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
    classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]
    # # Turn the data into a set of points:
    data = list(zip(x, y))
    # fit the KNN algorithm with K=1
    # knn = KNeighborsClassifier(n_neighbors=1)
    # do the same thing, but with a higher K value which changes the prediction:
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(data, classes)
    # use it to classify a new data point:
    new_x = 8
    new_y = 21
    new_point = [(new_x, new_y)]

    prediction = knn.predict(new_point)

    plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
    plt.text(x=new_x - 1.7, y=new_y - 0.7, s=f"new point, class: {prediction[0]}")

    plt.show()
    return


if __name__ == '__main__':
    print(standart_deviation())
    print(variance())
    print(percentile())
    print(data_set())
    gaussian_data_distribution()
    plot()
    random_data_distribution()
    linear_regression()
    polinomial_regression()
    print('Predicted CO2: ', multiple_regression(), 'grams')
    print('Predicted CO2: ', second_example_multiple_regression(), 'grams')
    print(scale_values())
    print(second_scaled_values())
    train_test()
    decision_tree()
    conf_matrix()
    hierarchical_clustering()
    print('predicted: ', logistic_regression_binomial())
    print(probability())
    default_c()
    print(grid_search())
    print(categorical_data_one_hot_encoding())
    print(dummifying())
    k_means()
    bagging()
    creating_bagging_classifier()
    print(out_of_bag_score())
    decision_tree_from_bagging_classifier()
    cross_validation()
    imbalanced_data()
    two_sets_of_probabilities()
    k_nearest_neighbors()
