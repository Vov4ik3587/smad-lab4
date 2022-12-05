import numpy as np
from scipy.stats import chi2, f


class Model:  # Модель из предыдущих лабораторных работ

    def __init__(self):
        self.amount_tests = 900
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
new_amount_tests = int(model.amount_tests / 3)

matrix_X = np.vstack([model.x1, model.x2, model.variance])
matrix_X_sort = matrix_X[matrix_X[:, 2].argsort()[::-1]]  # Сортируем по убыванию взвешенной суммы квадратов факторов

# Раделяем массив на 2 выборки
new_model_start = Model()
new_model_end = Model()

new_model_start.amount_tests, new_model_end.amount_tests = new_amount_tests, new_amount_tests
new_model_start.x1, new_model_start.x2 = matrix_X_sort[0][:new_amount_tests], matrix_X_sort[1][:new_amount_tests]
new_model_end.x1, new_model_end.x2 = matrix_X_sort[0][new_amount_tests * 2:], matrix_X_sort[1][new_amount_tests * 2:]

new_model_start.signal = Calculator.compute_signal(new_model_start)
new_model_end.signal = Calculator.compute_signal(new_model_end)

new_model_start.response = new_model_start.signal + matrix_X_sort[2][:new_amount_tests]
new_model_end.response = new_model_end.signal + matrix_X_sort[2][new_amount_tests * 2:]

# Оценим обе выборки по МНК
new_model_start.experiment_matrix = Calculator.compute_experiment_matrix(new_model_start)
new_model_end.experiment_matrix = Calculator.compute_experiment_matrix(new_model_end)

new_model_start.theta_mnk = Calculator.mnk(new_model_start)
new_model_end.theta_mnk = Calculator.mnk(new_model_end)

# Вычислим RSS для каждой выборки
RSS_start, RSS_end = 0, 0

for i in range(new_amount_tests):
    RSS_start += (new_model_start.response[i] -
                  np.matmul(new_model_start.experiment_matrix, new_model_start.theta_mnk)[i]) ** 2
    RSS_end += (new_model_end.response[i] - np.matmul(new_model_end.experiment_matrix, new_model_end.theta_mnk)[i]) ** 2

f_stat = f.ppf(0.95, 248, 248)

if RSS_end / RSS_start > f_stat:
    print('Гипотеза об отсутствии гетероскедастичности возмущений отвергается')
else:
    print('гипотеза об отсутствии гетероскедастичности не отвергается')


# %% Вычислим ОМНК и сравним с МНК

model.theta_general_mnk = Calculator.general_mnk(model)
print(f"МНК: {model.theta_mnk}")
print(f"ОМНК: {model.theta_general_mnk}")
print(np.square(model.theta_mnk - model.theta_general_mnk))