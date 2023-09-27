import numpy
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    print(standart_deviation())
    print(variance())
    print(percentile())
    print(data_set())
