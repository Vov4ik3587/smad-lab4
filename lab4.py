import numpy as np


class Model:  # Модель из предыдущих лабораторных работ

    def __init__(self):
        self.amount_tests = 12
        self.x_max = 1
        self.x_min = -1
        self.x1 = []
        self.x2 = []
        self.signal = []  # сигнал
        self.response = []  # отклик
        self.power = 0  # мощность сигнала
        self.variance = 0  # дисперсия
        self.theta = [1, 4, 0.001, 4]  # параметры модели
        self.func = lambda x1, x2: 1 + 4 * x1 + 0.001 * x1 ** 2 + 4 * x2 ** 2


class Calculator:

    @staticmethod
    def compute_signal(model: Model):  # Вычисление сигнала - незашумленного отклика
        signal = [model.func(model.x1[i], model.x2[i])
                  for i in range(model.amount_tests)]
        return np.array(signal)

    @staticmethod
    def compute_variance(model) -> float:  # Вычисление дисперсии
        return model.power * 0.1

    @staticmethod
    def compute_power(model):  # Вычисление мощности
        avg_signal = [
            np.sum(model.signal) / len(model.signal)
            for i in range(len(model.signal))
        ]
        vec_avg_signal = np.array(avg_signal)
        power = np.vdot(model.signal - vec_avg_signal,
                        model.signal - vec_avg_signal) / len(model.signal)
        return power

    @staticmethod
    def compute_response(model, error):  # вычисление зашумленного отклика
        return model.signal + error


class DataGenerator:

    @staticmethod
    def generate_couple(x_min, x_max, amount_tests):  # Генерация значений регрессоров
        x1 = np.random.uniform(x_min, x_max, amount_tests)
        x2 = np.random.uniform(x_min, x_max, amount_tests)
        return x1, x2

    @staticmethod
    def generate_error(standard_deviation, number_tests) -> float:  # генерация случайной ошибки
        error = np.random.normal(0, standard_deviation, number_tests)  # стандартное отклонение - sqrt(variance)
        return error
