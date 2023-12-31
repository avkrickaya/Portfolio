# Классификация отзывов

# Содержание
1. [Описание задачи](https://github.com/avkrickaya/Portfolio/blob/main/Классификация%20отзывов/README.md#описание-задачи)
2. [Описание данных](https://github.com/avkrickaya/Portfolio/blob/main/Классификация%20отзывов/README.md#описание-данных)
3. [Используемые инструменты](https://github.com/avkrickaya/Portfolio/blob/main/Классификация%20отзывов/README.md#используемые-инструменты)
4. [План работы](https://github.com/avkrickaya/Portfolio/blob/main/Классификация%20отзывов/README.md#план-работы)
5. [Вывод](https://github.com/avkrickaya/Portfolio/blob/main/Классификация%20отзывов/README.md#вывод)

## Описание задачи 

Интернет-магазин запускает новый сервис, в котором пользователи могут редактировать и дополнять описания товаров, как в вики-сообществах. То есть клиенты предлагают свои правки и комментируют изменения других. Магазину нужен инструмент, который будет искать токсичные комментарии и отправлять их на модерацию.

Требуется обучить модель классифицировать комментарии на позитивные и негативные. В нашем распоряжении набор данных с разметкой о токсичности правок.

## Описание данных

Данные находятся в файле toxic_comments.csv.

Столбец text в нём содержит текст комментария, а toxic — целевой признак.


## Используемые инструменты 

 Pandas, NLTK, Numpy, Wordcloud, Matplotlib, , Sklearn

## План работы

1. Первичное ознакомление, анализ данных и подготовка к обучению:
   
     1.1. Анализ частоты слов
   
     1.2. Тематика отзывов
   
     1.3. Учет дисбаланса данных
   
2. Анализ моделей
3. Определение порога классикации модели
4. Тестирование
5. Вывод


## Вывод

В процессе решения поставленной задачи были сделаны следующие действия:


•	Данные были загружены, очищены от лишних знаков и слов без смысловой нагрузки, а также лемматизированы.  В датафрейме содержится 159 292 строк и 2 столбца. Пропусков в данных не обнаружено.


•	Проведен анализ частоты слов в негативных и положительных слов. Есть слова, которые присутствуют и в положительных, и в отрицательных. Наиболее часто употребляемые слова все же имеют различия в зависимости от эмоциональной окраски отзыва.


•	Все отзывы были разделены на 4 тематики.


•	В процессе предобработки и анализа отзывов был выявлен дисбаланс данных. Для борьбы с этим использовался метод учета веса классов, что позволило повысить значение метрики примерно на 0,05.


•	Данные были разделены на обучающую и тестовую выборки в соотношении 3:1 (75%/25%). Выделены признаки и целевой.


•	Для исследования были отобраны три модели, которые показали следующие результаты: 


|                           | F1 на CV | AUC-ROC  |
|---------------------------|----------|----------|
| LogisticRegression        | 0.768754 | 0.971793 |
| DecisionTreeClassifier    | 0.528904 | 0.796716 |
| SGDClassifier             | 0.750357 | 0.971877 |
                                        

•	Наилучшие показатели метрики f1 у модели LogisticRegression. На втором месте SGDClassifier. Наихудшие показатели метрики f1 у модели DecisionTreeClassifier.  Тестирование проводилось на модели LogisticRegression. Перед тестирование был определен порог классификации для этой модели и построен график Точность-Полнота. Наиболее подходящим пороговым значением является 0.72.


•	Метрика f1 на тестовой выборке с учетом подобранных параметров и учета порога классификации показала значение выше требуемого (>0.75), значит данную модель мы можем использовать для дальнейшей работы. 


    F1 на тестовой выборке 0.7892967942088934

