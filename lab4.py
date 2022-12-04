import numpy as np
from scipy.stats import chi2


class Model:  # Модель из предыдущих лабораторных работ

    def __init__(self):
        self.amount_tests = 500
        self.x_max = 1
        self.x_min = -1
        self.x1 = []
        self.x2 = []
        self.signal = []  # сигнал
        self.response = []  # отклик
        self.variance = []  # дисперсия
        self.theta = np.array([1, 4, 0.001, 4])  # параметры модели
        self.theta_mnk = []  # Оценка теты по МНК
        self.theta_general_mnk = []  # Оценка теты по обобщенному МНК
        self.func = lambda x1, x2: 1 + 4 * x1 + 0.001 * x1 ** 2 + 4 * x2 ** 2
        self.experiment_matrix = []


class Calculator:

    @staticmethod
    def compute_signal(model: Model):  # Вычисление сигнала - незашумленного отклика
        signal = [model.func(model.x1[i], model.x2[i])
                  for i in range(model.amount_tests)]
        return np.array(signal)

    @staticmethod
    def compute_variance(model):  # Вычисление дисперсии (взвешенная сумма квадратов факторов)
        result = np.array([0.5 * model.x1[i] ** 2 + 0.5 * model.x2[i] ** 2 for i in range(model.amount_tests)])
        return result

    @staticmethod
    def compute_response(model, error):  # вычисление зашумленного отклика
        return model.signal + error

    @staticmethod
    def general_mnk(model):  # Обобщенный метод наименьших квадратов
        matrix_V = np.diag(model.variance)
        general_mnk_eval = np.matmul(np.matmul(np.matmul(np.linalg.inv(
            np.matmul(np.matmul(model.experiment_matrix.T, np.linalg.inv(matrix_V)), model.experiment_matrix)),
            model.experiment_matrix.T), np.linalg.inv(matrix_V)), model.response)
        return general_mnk_eval

    @staticmethod
    def mnk(model):  # Метод наименьших квадратов
        trans_experiment_matrix = model.experiment_matrix.T
        mnk_eval = np.matmul(np.linalg.inv(np.matmul(trans_experiment_matrix, model.experiment_matrix)),
                             trans_experiment_matrix)
        mnk_eval = np.matmul(mnk_eval, model.response)
        return mnk_eval

    @staticmethod
    def compute_experiment_matrix(model):  # Матрица наблюдений X
        experiment_matrix = np.array([
            np.array([1 for _ in range(model.amount_tests)]),
            model.x1,
            np.array([x1 ** 2 for x1 in model.x1]),
            np.array([x2 ** 2 for x2 in model.x2])
        ], dtype=object)
        experiment_matrix = np.array([list(i) for i in zip(*experiment_matrix)])
        return experiment_matrix


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


# %% Заполняем модель данными

model = Model()

model.x1, model.x2 = DataGenerator.generate_couple(
    model.x_min, model.x_max, model.amount_tests)

model.signal = Calculator.compute_signal(model)
model.variance = Calculator.compute_variance(model)

error = DataGenerator.generate_error(
    np.sqrt(model.variance), model.amount_tests)

model.response = Calculator.compute_response(model, error)

model.experiment_matrix = Calculator.compute_experiment_matrix(model)

# model.theta_general_mnk = Calculator.general_mnk(model)

# %% Проверка данных на гетероскедастичность. Тест Бреуша-Пагана

model.theta_mnk = Calculator.mnk(model)  # Оцениваем тету по МНК
residual_vec = model.response - np.matmul(model.experiment_matrix, model.theta_mnk)  # Находим вектор остатков
variance_eval = np.sum(np.square(residual_vec)) / model.amount_tests  # Оцениваем дисперсию

new_response = np.array([elem ** 2 / variance_eval for elem in residual_vec])  # отклик для новой регрессии

vec_z = np.ones(model.amount_tests)  # Вектор известных факторов
experiment_matrix_z = np.vstack([vec_z, model.variance]).T  # Матрица наблюдений Z
alpha_eval = np.matmul(
    np.matmul(np.linalg.inv(np.matmul(experiment_matrix_z.T, experiment_matrix_z)), experiment_matrix_z.T),
    new_response)  # Оцениваем неизвестные параметры альфа

new_response_hat = np.matmul(alpha_eval, experiment_matrix_z.T)  # c с крышечкой
avg_new_response = np.mean(new_response)  # среднее арифметическое нового отклика
ESS = np.sum(np.square(np.subtract(new_response_hat, avg_new_response)))  # Объясненная сумма квадратов
chi_stat = chi2.ppf(0.95, 1)  # квантиль Хи^2

if ESS / 2 > chi_stat:
    print('Гипотеза об отсутствии гетероскедастичности возмущений отвергается')
else:
    print('Гипотеза об отсутствии гетероскедастичности возмущений не отвергается')

# %% Проверка данных на гетероскедастичность. Тест Голдфельда-Квандтона

