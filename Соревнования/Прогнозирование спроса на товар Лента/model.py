def forecast(sales: dict, item_info: dict, store_info: dict) -> list:
    """
    Функция для предсказания продаж
    :params sales: исторические данные по продажам
    :params item_info: характеристики товара
    :params store_info: характеристики магазина

    """
    # Импорт библиотек
    # Импорт библиотек
    import pandas as pd
    import numpy as np
    import warnings
    import pickle
    from statsmodels.tsa.seasonal import seasonal_decompose
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    warnings.filterwarnings('ignore')

    # Создание Dataframe df из полученных словарей, объединение их в 1 и удаление лишних данных
    train = pd.DataFrame.from_dict('sales')
    pr = pd.DataFrame.from_dict('item_info')
    st_df = pd.DataFrame.from_dict('store_info')
    holiday = pd.read_csv('holidays_covid_calendar.csv')

    # Приведение формата даты к нужному формату
    train['date'] = pd.to_datetime(train['date'])
    holiday['date'] = pd.to_datetime(holiday['date'], format="%d.%m.%Y", infer_datetime_format=True)

    temp_df = pd.merge(train, pr, on='pr_sku_id')
    df = pd.merge(temp_df, st_df, on='st_id')
    del temp_df

    # Закодируем хэши в числа для удобства работы моделей
    hash_to_num_dict = {}
    num_to_hash_dict = {}

    for column in df.columns:
        if df[column].dtype == 'object':  # Проверяем, является ли столбец хэшированным
            unique_values = df[column].unique()
            hash_to_num = {value: num for num, value in enumerate(unique_values)}
            num_to_hash = {num: value for num, value in enumerate(unique_values)}
            hash_to_num_dict[column] = hash_to_num
            num_to_hash_dict[column] = num_to_hash

    df_num = df.copy()
    for column, hash_to_num in hash_to_num_dict.items():
        df_num[column] = df[column].map(hash_to_num)

    # Удалить отрицательные значения в столбцах 'pr_sales_in_units', 'pr_promo_sales_in_units',
    # 'pr_sales_in_rub' и 'pr_promo_sales_in_rub'
    df_num = df_num.loc[(df_num['pr_sales_in_units'] >= 0) & (df_num['pr_promo_sales_in_units'] >= 0) &
                        (df_num['pr_sales_in_rub'] >= 0) & (df_num['pr_promo_sales_in_rub'] >= 0)]

    # Удалить строки с нулевыми продажами, но ненулевой прибыль 
    df_num = df_num.query('pr_sales_in_units > 0')

    # Заполним товары, у которых сумма продаж равна нулю произведением продаж товара в штуках на стоимость товара.
    df_num['price'] = round(df_num['pr_sales_in_rub'] / df_num['pr_sales_in_units'], 2)
    mean_price = df_num.groupby('pr_sku_id')['pr_sales_in_rub'].mean()

    zero_sales = df_num['pr_sales_in_rub'] == 0.0
    # Замена значения 0.0 в столбце price и pr_sales_in_rub
    df_num.loc[zero_sales, 'price'] = df_num.loc[zero_sales, 'pr_sku_id'].map(mean_price)
    df_num.loc[zero_sales, 'pr_sales_in_rub'] = \
        df_num.loc[zero_sales, 'pr_sku_id'].map(mean_price) * df_num.loc[zero_sales, 'pr_sales_in_units']

    # Удалить все строки, где st_is_active == 0
    df_num = df_num[df_num['st_is_active'] != 0]

    # Удалить столбец 'st_is_active' за ненадобностью
    df_num = df_num.drop(columns=['st_is_active'])

    # У нас представлено очень маленькое количество товаров 7 и 8 группы.
    # Такие группы следует исключить из дальнейшего исследования.
    df_num = df_num.query('st_id != (7, 11)')
    df_num = df_num.query('pr_group_id != (7, 8)')

    # Сортировака даты по возрастанию
    df_num = df_num.sort_values('date')

    # Генерация фичей
    def generate_features(df_num):
        df_num = df_num.sort_values('date', ascending=False)
        df_num = df_num.assign(
            pr_sales_in_units_lag=df_num.groupby('pr_cat_id')['pr_sales_in_units'].shift(14),
            pr_sales_in_rub_lag=df_num.groupby('pr_cat_id')['pr_sales_in_rub'].shift(14),
            pr_sales_in_units_max_lag=df_num.groupby('pr_cat_id')['pr_sales_in_units'].transform(
                lambda x: x.rolling(window=14, min_periods=1).max()),
            pr_sales_in_units_min_lag=df_num.groupby('pr_cat_id')['pr_sales_in_units'].transform(
                lambda x: x.rolling(window=14, min_periods=1).min()),
            pr_sales_in_rub_max_lag=df_num.groupby('pr_cat_id')['pr_sales_in_rub'].transform(
                lambda x: x.rolling(window=14, min_periods=1).max()),
            pr_sales_in_rub_min_lag=df_num.groupby('pr_cat_id')['pr_sales_in_rub'].transform(
                lambda x: x.rolling(window=14, min_periods=1).min()),
            mean_sales_week_lag=df_num.groupby(['pr_cat_id', 'pr_sku_id'])['pr_sales_in_units'].transform(
                lambda x: x.rolling(window=14).mean()),
            lag_feature_weekday=df_num['date'].dt.weekday.shift(1),
            month=df_num['date'].dt.month,
            quarter=df_num['date'].dt.quarter,
            lag_feature_4weeks=df_num['pr_sales_in_units'].rolling(window=29).sum(),
            lag_feature_1week=df_num['pr_sales_in_units'].rolling(window=14).sum(),
            sales_ratio=lambda x: x['lag_feature_4weeks'] / x['lag_feature_1week'],
            sales_ratio_cat=df_num['pr_sales_in_units'] / df_num.groupby('pr_cat_id')['pr_sales_in_units'].transform(
                'sum'),
            lag_dayofyear=df_num['date'].dt.dayofyear,
            sales_slope_7d=df_num['pr_sales_in_units'].rolling(window=14).apply(
                lambda x: np.polyfit(range(14), x, 1)[0], raw=True),
            pr_sales_in_units_lag_city=df_num.groupby('st_city_id')['pr_sales_in_units'].shift(14),
            pr_sales_in_rub_lag_city=df_num.groupby('st_city_id')['pr_sales_in_rub'].shift(14),
            pr_sales_in_units_lag_local_type=df_num.groupby('st_type_loc_id')['pr_sales_in_units'].shift(14),
            pr_sales_in_rub_lag_local_type=df_num.groupby('st_type_loc_id')['pr_sales_in_rub'].shift(14),
            seasonality_index=df_num.groupby('st_id')['pr_sales_in_units'].transform(lambda x: x / x.mean())
        )

        df_num.set_index('date', inplace=True)

        for i in range(1, 15):
            df_num[f'units_lag_{i}_units'] = df_num.groupby('pr_cat_id')['pr_sales_in_units'].shift(i)

        df_num = df_num.dropna()

        return df_num

    df_num = generate_features(df_num)

    # Добавление дат праздников, удалиние лишних столбцов
    df_num = pd.merge(df_num, holiday, on='date').drop(['year', 'day', 'weekday', 'calday', 'covid'], axis=1)

    # Загрузить обученные модели
    ls_model = pickle.loads('Lasso.pkl')
    xgb_model = pickle.loads('XGBRegressor.pkl')
    lgbm_model = pickle.loads('LGBMRegressor.pkl')
    dr_model = pickle.loads('DecisionTreeRegressor.pkl')
    meta_model = pickle.loads('meta_model_units.pkl')
    all_models = [ls_model, xgb_model, lgbm_model, dr_model]

    # Предсказания
    # Функция для получения предсказаний и подготовки результатов к пролучению предсказаний мета-модели
    def predict_model_1st_level(df, all_models):

        df = generate_features(df)

        features_df = df.drop('pr_sales_in_units', axis=1)

        meta_feat = []
        for model in all_models:
            y_pred = model.predict(features_df)
            meta_feat.append(y_pred)

        meta_feat_array = np.array(meta_feat, dtype=np.float32).T.reshape(-1, len(meta_feat))

        return meta_feat_array, features_df

    meta_feat_array_test, features_df_test = predict_model_1st_level(df_num, all_models)

    meta_predict_test = meta_model.predict(meta_feat_array_test)

    predict_df = pd.DataFrame(meta_predict_test, columns=['target'])
    predict_df = round(predict_df)

    features_df_test.reset_index(inplace=True)

    df_predictions = pd.DataFrame({'st_id': features_df_test['st_id'],
                                   'pr_sku_id': features_df_test['pr_sku_id'],
                                   'date': features_df_test['date']})

    # Замена порядковых номеров на исходные значения хэшей
    df_predictions['st_id'] = df_predictions['st_id'].map(num_to_hash_dict['st_id'])
    df_predictions['pr_sku_id'] = df_predictions['pr_sku_id'].map(num_to_hash_dict['pr_sku_id'])

    # Создание датафрейма с исходными значениями и предсказаниями
    sales_submission = pd.concat([df_predictions[['st_id', 'pr_sku_id', 'date']], predict_df], axis=1)

    return sales_submission
