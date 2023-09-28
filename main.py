import numpy
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model, tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def standart_deviation():
    speed = [40, 42, 59, 70, 113, 37, 54, 88, 32, 47, 50]
    std_speed = numpy.std(speed)
    return std_speed


def variance():
    speed = [40, 42, 59, 70, 113, 37, 54, 88, 32, 47, 50]
    var_speed = numpy.var(speed)
    return var_speed


def percentile():
    """ What is the age that 90% of the people are younger than? """

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