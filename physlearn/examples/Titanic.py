import math
import sys
import os
from list2vector import list2vector, normalize_vector

import numpy
import pandas as pd


def create_datasets(cv_percent):
    module = str(sys.modules['physlearn.examples'])
    split_path = module.split('\'')
    dir_path = os.path.split(split_path[3])
    os.chdir(dir_path[0])
    train_data = pd.read_csv('data/train.csv')

    class_list = list(train_data['Pclass'])  # класс каюты
    sex_list = list(train_data['Sex'])  # пол
    age_list = list(train_data['Age'])  # возраст
    survived_list = list(train_data['Survived'])  # выжил или нет

    remove_list = []
    for index, item in enumerate(age_list):
        if math.isnan(item):  # Проверка на NaN
            remove_list.append(index)  # Если NaN - добавляем индекс элемента в список
    remove_list.reverse()  # Обращаем список, что бы удаление шло с конца и не менялись индексы элементов,
    # подлежащих удалению
    for index in remove_list:  # удаляем элементы
        age_list.pop(index)
        sex_list.pop(index)
        class_list.pop(index)
        survived_list.pop(index)

    # Переводим списки в числовой вид, превращаем в массивы и нормализуем (за исключение survived_list,
    # его только в массив)
    class_array = normalize_vector(list2vector(class_list))
    sex_array = normalize_vector(list2vector(sex_list))
    age_array = normalize_vector(list2vector(age_list))
    survived_array = list2vector(survived_list)

    # Разделяем данные на обучающую и проверочные выборки
    total_len = class_array.shape[0]
    break_point = math.floor((1 - cv_percent) * total_len)

    class_learn = class_array[:break_point]
    class_cv = class_array[break_point:]

    sex_learn = sex_array[:break_point]
    sex_cv = sex_array[break_point:]

    age_learn = age_array[:break_point]
    age_cv = age_array[break_point:]

    # Создаем выходные данные
    learn_output = numpy.array([survived_array[:break_point]])
    cv_output = numpy.array([survived_array[break_point:]])

    # Создаем входные матрицы
    learn_data = numpy.array([class_learn, sex_learn, age_learn])
    cv_data = numpy.array([class_cv, sex_cv, age_cv])

    return (learn_data, learn_output), (cv_data, cv_output)
