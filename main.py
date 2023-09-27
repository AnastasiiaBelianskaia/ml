import numpy
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score


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


if __name__ == '__main__':
    # print(standart_deviation())
    # print(variance())
    # print(percentile())
    # print(data_set())
    # gaussian_data_distribution()
    # plot()
    # random_data_distribution()
    # linear_regression()
    polinomial_regression()
