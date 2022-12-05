# Статистические методы анализа данных лабораторная работа 4
# Вариант №6 Бригада:Абраменко, Мак, Назаров

import numpy as np
from numpy.linalg import inv
from scipy.stats import f, chi2

np.random.seed(1)
N = 600  # Кол-во наблюдений
theta = [1, 1, 1, 1, 0.01, 0.01, 0.01, 1, 1]  # Истинный вектор тетта


# Генерация N наблюдений
def gen_x():
    x_values = np.linspace(-1, 1, 21)
    x1 = np.random.choice(x_values, N)
    x2 = np.random.choice(x_values, N)
    x3 = np.random.choice(x_values, N)
    return x1, x2, x3


# Вспомогательная функция для вычисления суммы взвешенных квадратов факторов
def wss(x1, x2, x3):
    return 0.3*x1**2 + 0.3*x2**2 + 0.3*x3**2


# Вспомогательная функция для вычисления незашумленного отклика
def func(thetta, x1, x2, x3):
    return thetta[0] + thetta[1]*x1 + thetta[2]*x2 + thetta[3]*x3 + \
           thetta[4]*x1**2 + thetta[5]*x2**2 + thetta[6]*np.cos(x3) +\
           thetta[7]*x1*x2 + thetta[8]*x1*x3


# Обобщённый метод наименьших квадратов
def omnk(data_print):
    x1, x2, x3 = gen_x()
    X, y, theta_est = gen_param()
    V = np.zeros([N, N])
    for i in range(N):
        V[i][i] = wss(x1[i], x2[i], x3[i])

    theta_est_omnk = np.dot(np.dot(np.dot(inv(np.dot(np.dot(X.transpose(), inv(V)), X)), X.transpose()), inv(V)), y)

    for i in range(9):
        print((theta[i]-theta_est_omnk[i])**2)

    if data_print:
        print("Оценочные значения тетта по МНК:", theta_est)
        print("Оценочные значения тетта по ОМНК:", theta_est_omnk)


def gen_param():

    x1, x2, x3 = gen_x()
    e, u, y = np.empty(N), np.empty(N), np.empty(N)

    for i in range(N):
        u[i] = func(theta, x1[i], x2[i], x3[i])   # Незашумленный отклик
        # e[i] - ошибка с дисперсией, равной экспоненте от взвешенной суммы квадратов факторов
        e[i] = np.random.normal(0, np.exp(wss(x1[i], x2[i], x3[i])))
        y[i] = u[i] + e[i]  # Зашумленный отклик

    # Оцениваем модель по МНК
    z = np.ones(N)  # Вспомогательный вектор из единичек
    x1x2, x1x3, x1_square, x2_square, x3_cos = np.empty(N), np.empty(N), np.empty(N), np.empty(N), np.empty(N)
    for i in range(N):
        x1x2[i], x1x3[i] = x1[i] * x2[i], x1[i] * x3[i]
        x1_square[i], x2_square[i], x3_cos[i] = x1[i] ** 2, x2[i] ** 2, np.cos(x3[i])

    # Матрица X
    X = np.vstack([z, x1, x2, x3, x1_square, x2_square, x3_cos, x1x2, x1x3]).transpose()

    # Оценочные значения тетта
    theta_est = np.dot(np.dot(inv(np.dot(X.transpose(), X)), X.transpose()), y)

    return X, y, theta_est


# Тест Бреуша-Пагана
def breusch_pagan():
    disp_est, ESS = 0, 0
    c, ewss = np.empty(N), np.empty(N)
    x1, x2, x3 = gen_x()
    X, y, theta_est = gen_param()

    remains = y - np.dot(X, theta_est)  # остатки

    # Оценивание дисперсии
    for i in range(N):
        disp_est += remains[i]**2 / N
    # Новый отклик для регрессии
    for i in range(N):
        c[i] = remains[i] ** 2 / disp_est
        ewss[i] = np.exp(wss(x1[i], x2[i], x3[i]))

    z = np.ones(N)
    Xz = (np.vstack([z, ewss])).transpose()
    # Оценочные значения регрессоров (1,z1,....,zm)
    alpha_est = np.dot(np.dot(inv(np.dot(Xz.transpose(), Xz)), Xz.transpose()), c)
    # Отклик от оценочных значений параметров и его среднее значение
    c_ = np.dot(alpha_est, Xz.transpose())
    c_avg = np.mean(c)

    for i in range(N):
        ESS += (c_[i]-c_avg)**2  # Объяснённая сумма квадратов связанная с ре-грессорами alpha_est

    chi_stat = chi2.ppf(0.95, 1)
    print('квантиль хи2 = ', chi_stat)
    print("ECC/2 = ", ESS / 2)
    if ESS / 2 > chi_stat:
        print('гипотеза о гомоскедастичности возмущений отвергается')
    else:
        print('гипотеза о гомоскедастичности возмущений не отвергается')


# Тест Голдфельда-Квандтона
def goldfeld_quandt():
    NC = int(N/3)
    rss_up, rss_down = 0, 0
    x1, x2, x3 = gen_x()
    ewss = np.empty(N)
    u_up, u_down, y_up, y_down = np.empty(NC), np.empty(NC), np.empty(NC), np.empty(NC)
    x1_up, x2_up, x3_up = np.empty(NC), np.empty(NC), np.empty(NC)
    x1_down, x2_down, x3_down = np.empty(NC), np.empty(NC), np.empty(NC)
    for i in range(N):
        ewss[i] = np.exp(wss(x1[i], x2[i], x3[i]))

    # Сортируем массив по убыванию взвешенной суммы квадратов
    X_sort = np.vstack([x1, x2, x3, ewss]).transpose()
    X_sort = X_sort[X_sort[:, 3].argsort()[::-1]]
    # Разделяем полученный массив на две выборки: N/3 первых и последних наблю-дений
    x_up = X_sort[:200]
    x_down = X_sort[400:]

    # Оцениваем эти выборки по МНК
    for i in range(200):
        x1_up[i], x2_up[i], x3_up[i] = x_up[i][0], x_up[i][1], x_up[i][2]
        x1_down[i], x2_down[i], x3_down[i] = x_down[i][0], x_down[i][1], x_down[i][2]
        u_up[i] = func(theta, x_up[i][0], x_up[i][1], x_up[i][2])
        u_down[i] = func(theta, x_down[i][0], x_down[i][1], x_down[i][2])
        y_up[i] = u_up[i] + x_up[i][3]
        y_down[i] = u_down[i] + x_down[i][3]

    # Вычисление остаточной суммы квадратов для верхней выборки
    z = np.ones(200)  # Вспомогательный вектор из единичек
    x1x2, x1x3, x1_square, x2_square, x3_cos = np.empty(NC), np.empty(NC), np.empty(NC), np.empty(NC), np.empty(NC)
    for i in range(200):
        x1x2[i], x1x3[i] = x_up[i][0] * x_up[i][1], x_up[i][0] * x_up[i][2]
        x1_square[i], x2_square[i], x3_cos[i] = x_up[i][0] ** 2, x_up[i][1] ** 2, np.cos(x_up[i][2])

    X_up = np.vstack([z, x1_up, x2_up, x3_up, x1_square, x2_square, x3_cos, x1x2, x1x3]).transpose()

    # Оценочные значения параметров для верхней выборки
    theta_est_up = np.dot(np.dot(inv(np.dot(X_up.transpose(), X_up)), X_up.transpose()), y_up)
    for i in range(200):
        rss_up += (y_up[i] - np.dot(X_up, theta_est_up)[i])**2

    # Вычисление остаточной суммы квадратов для нижней выборки

    for i in range(200):
        x1x2[i], x1x3[i] = x_down[i][0] * x_down[i][1], x_down[i][0] * x_down[i][2]
        x1_square[i], x2_square[i], x3_cos[i] = x_down[i][0] ** 2, x_down[i][1] ** 2, np.cos(x_down[i][2])

    X_down = np.vstack([z, x1_down, x2_down, x3_down, x1_square, x2_square, x3_cos, x1x2, x1x3]).transpose()
    # Оценочные значения параметров для нижней выборки
    theta_est_down = np.dot(np.dot(inv(np.dot(X_down.transpose(), X_down)), X_down.transpose()), y_down)
    for i in range(200):
        rss_down += (y_down[i] - np.dot(X_down, theta_est_down)[i])**2

    f_stat = f.ppf(0.95, 248, 248)
    print('RSS UP = ', rss_up)
    print('RSS DOWN = ', rss_down)
    print('квантиль F = ', f_stat)
    print("rss_up/rss_down = ", rss_down/rss_up)
    if rss_down/rss_up > f_stat:
        print('гипотеза о гомоскедастичности возмущений отвергается')
    else:
        print('гипотеза о гомоскедастичности возмущений не отвергается')


def main():
    data_print = True
    # Функции запускать по одной, иначе где-то происходит утечка данных при вы-числении ОМНК
    breusch_pagan()
    goldfeld_quandt()
    omnk(data_print)


main()
