# Определение возраста покупателей

## Содержание:
1.	[Описание проекта](https://github.com/avkrickaya/Portfolio/blob/main/Определение%20возраста%20покупателей/README.md#описание-проекта)
2.	[Условия](https://github.com/avkrickaya/Portfolio/blob/main/Определение%20возраста%20покупателей/README.md#условия)
3.	[Описание данных](https://github.com/avkrickaya/Portfolio/blob/main/Определение%20возраста%20покупателей/README.md#описание-данных)
4.	[Используемые инструменты](https://github.com/avkrickaya/Portfolio/blob/main/Определение%20возраста%20покупателей/README.md#используемые-инструменты)
5.	[План работы](https://github.com/avkrickaya/Portfolio/blob/main/Определение%20возраста%20покупателей/README.md#план-работы)
6.	[Вывод](https://github.com/avkrickaya/Portfolio/blob/main/Определение%20возраста%20покупателей/README.md#вывод)


## Описание проекта

Сетевой супермаркет внедряет систему компьютерного зрения для обработки фотографий покупателей. Фотофиксация в прикассовой зоне поможет определять возраст клиентов, чтобы:

•	Анализировать покупки и предлагать товары, которые могут заинтересовать покупателей этой возрастной группы;

•	Контролировать добросовестность кассиров при продаже алкоголя.

•	Постройте модель, которая по фотографии определит приблизительный возраст человека. В вашем распоряжении набор фотографий людей с указанием возраста.

## Условия

Для заказчика важно, чтобы метрика MAE < 8.


## Описание данных

•	file_name - наименование файла
•	real_age - реальный фозраст покупателя.

## Используемые инструменты
Tensorflow, Keras, Seaborn, Matplotlib, Pandas


## План работы

1.	Исследовательский анализ данных. Загрузить и изучить данные. Проверить распределение, наличие аномалий, выявление взаимосвязей.
	
2.	Обучение модели
	
3.	Анализ модели. Описание полученных результатов.


## Вывод
•	Был загружен и изучен датафрейм с фотографиями и возрастом людей на них. В нем содержится 7591 строк и 2 столбца. Пропуски отсутствуют.


•	Наиболее распространённый возраст людей на фотографиях находится в диапазоне от 20 до 40 лет. Присутствуют аномальные выбросы начиная с возраста больше 70 лет. Это можно объяснить, например, продолжительностью жизни людей. Также есть провал в возрасте 10-15 лет.


•	Фотографии имеют одинаковый размер, отсутствуют перевернутые фотографии. Качество фотографий доступных для обучения разное. Не все фотографии расположены ровно, так же присутствуют предметы, которые могут сказаться на обучении и снизить качество. Например, на некоторых фото у людей одеты очки или кепка. Так же не все смотрят в кадр, а некоторые фотографии имеют затемнение. Маленький ребенок на фотографии немного закрывает лицо рукой, что тоже может сказаться в дальнейшем. Еще среди фотографий присутствует черно-белая фотография.


•	Обучение модели проводилось в отдельном GPU-тренажёре, поэтому оформлен не как ячейка с кодом, а как код в текстовой ячейке.


•	По условию заказчика необходимо было построить нейронную сеть для определения возраста покупателей по фотографии. Для решения поставленной задачи была построена нейросете на базе предобученной сети ResNet50 с добавлением слоя пулинга и полносвязного слоя для классификации с одним нейроном. Для оптимизации дополнительно использовали алгоритм Adam с заданием скорости обучения 0.0001. Обучение проводилось на 5 эпохах.


•	Исходя из требований метрика качества была использована MAE. Значение метрики должно быть не выше 8. Результат нашей модели составляет 6,35. То есть нейросеть может выдавать ошибку при определении возраста +- 6,5 лет.

