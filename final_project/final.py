import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def find_department(data, store=24, dept=1, drop_list=[]):
    data_store = data[data['Store'] == store]
    data_store_dept = data_store[data_store['Dept'] == dept]
    data_store_dept['Date'] = pd.to_datetime(data_store_dept['Date'], format='%d-%m-%y')
    data_store_dept = data_store_dept.drop(drop_list, axis=1)
    data_store_dept = data_store_dept.reset_index()
    data_store_dept = data_store_dept.drop('index', axis=1)

    return data_store_dept


def split_data(data):
    data_train = data[data['Date'] < '01-01-11']
    data_val1 = data[data['Date'] >= '01-01-11']
    data_val = data_val1[data['Date'] < '01-01-12']
    data_test = data[data['Date'] >= '01-01-12']

    X_train = data_train.drop(['Weekly_Sales'], axis=1)
    X_train = X_train.set_index('Date')
    X_train.index = pd.to_datetime(X_train.index)
    y_train = data_train[['Date', 'Weekly_Sales']]
    y_train = y_train.set_index('Date')

    X_val = data_val.drop(['Weekly_Sales'], axis=1)
    X_val = X_val.set_index('Date')
    X_val.index = pd.to_datetime(X_val.index)
    y_val = data_val[['Date', 'Weekly_Sales']]
    y_val = y_val.set_index('Date')

    X_test = data_test.drop(['Weekly_Sales'], axis=1)
    X_test = X_test.set_index('Date')
    X_test.index = pd.to_datetime(X_test.index)
    y_test = data_test[['Date', 'Weekly_Sales']]
    y_test = y_test.set_index('Date')

    return X_train, X_val, X_test, y_train, y_val, y_test


def func(X_train, X_val, X_test, y_train, y_val, y_test, models_name='LinearRegression', store=24, dept=1,
         plot_test=False, val=False, par1='', par2='', par3=0, par4=0):
    print('Store ' + str(store) + ', Dept ' + str(dept))

    date_train = X_train.index
    date_val = X_val.index
    date_test = X_test.index

    scaler = MinMaxScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train))
    X_train_sc = X_train_sc.set_index(date_train)
    X_val_sc = pd.DataFrame(scaler.transform(X_val))
    X_val_sc = X_val_sc.set_index(date_val)
    X_test_sc = pd.DataFrame(scaler.transform(X_test))
    X_test_sc = X_test_sc.set_index(date_test)

    if models_name == 'LinearRegression':
        model = LinearRegression()

    elif models_name == 'DecisionTreeRegressor':
        # criterion='squared_error',    'squared_error', 'friedman_mse', 'absolute_error', 'poisson'
        # splitter='best',              'best', 'random'
        model = DecisionTreeRegressor(criterion=par1, splitter=par2)

    elif models_name == 'RandomForestRegressor':
        # criterion='squared_error',    'squared_error', 'friedman_mse', 'absolute_error', 'poisson'
        # n_estimators=100,             50, 100, 200, 300, 400, 500
        model = RandomForestRegressor(criterion=par1, n_estimators=par3)

    elif models_name == 'KNeighborsRegressor':
        # n_neighbors=10                range(2, 21)
        # weights='uniform'             'uniform', 'distance'
        # metric='manhattan'            'manhattan', 'euclidean', 'cityblock'
        model = KNeighborsRegressor(n_neighbors=par3, weights=par1, metric=par2)

    elif models_name == 'GradientBoostingRegressor':
        # loss='squared_error',         'squared_error', 'absolute_error'
        # n_estimators=100,             50, 100, 200, 300, 400, 500
        # learning_rate=0.1,            0.01, 0.05, 0.1, 0.5, 1
        model = GradientBoostingRegressor(n_estimators=par4, learning_rate=par3, loss=par1)

    if val:
        model.fit(X_train_sc, y_train)
        y_val_predict = model.predict(X_val_sc)

        mape = mean_absolute_percentage_error(y_val_predict, y_val)
        mae = mean_absolute_error(y_val_predict, y_val)

        print('MAPE (Validation Set): ' + str(mape))
        print('MAE (Validation Set): ' + str(mae))

        return model

    else:
        X_train_val_sc = pd.concat([X_train_sc, X_val_sc])
        y_train_val = pd.concat([y_train, y_val])
        model.fit(X_train_val_sc, y_train_val)

        test_results = X_test.copy()
        test_results = pd.concat([test_results, y_test['Weekly_Sales']], axis=1)
        test_results['prediction'] = model.predict(X_test_sc)

        test_plot = pd.DataFrame(test_results[['Weekly_Sales', 'prediction']])

        print('MAPE (Test Set): ' + str(mean_absolute_percentage_error(test_plot['prediction'], test_plot['Weekly_Sales'])))
        print('MAE (Test Set): ' + str(mean_absolute_error(test_plot['prediction'], test_plot['Weekly_Sales'])))

        if plot_test:
            test_plot.plot()
            plt.title('Test Set (Store '+str(store)+', Dept '+str(dept)+')')
            # plt.show()
            plt.savefig('results_'+models_name+'_store'+str(store)+'_dept'+str(dept)+'_test.png')

        # -- Predict Walmart Dataset --
        X = pd.concat([X_train, X_val, X_test])
        X_sc = pd.concat([X_train_sc, X_val_sc, X_test_sc])
        y = pd.concat([y_train, y_val, y_test])

        results = pd.concat([X, y['Weekly_Sales']], axis=1)
        results['prediction'] = model.predict(X_sc)

        prediction = results['prediction'].iloc[1:]
        prediction = prediction.tolist()
        prediction.append(np.nan)

        results2 = pd.DataFrame({'Date': y['Weekly_Sales'].index, 'Next Week': prediction})
        results2 = results2.set_index('Date')
        results = pd.concat([results, results2], axis=1)

        results_plot = pd.DataFrame(results[['Weekly_Sales', 'prediction']])
        if plot_test:
            results_plot.plot()
            plt.title('Walmart Dataset (Store '+str(store)+', Dept '+str(dept)+')')
            #plt.show()
            plt.savefig('results_'+models_name+'_store'+str(store)+'_dept'+str(dept)+'_dataset.png')

        results = results.drop(['prediction'], axis=1)

        results.to_excel('results_'+models_name+'_store'+str(store)+'_dept'+str(dept)+'.xlsx')

        return model, results


if __name__ == '__main__':
    # Store = 27
    # Dept 5, 8, 10

    walmart_data = pd.read_csv('walmart_cleaned.csv')
    data = walmart_data.drop(['Unnamed: 0', 'Next week'], axis=1)
    data = data.dropna()

    # drop_list = ['Store', 'Dept', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Type', 'Size']
    drop_list = ['Store', 'Dept', 'Type', 'Size']
    name = ['LinearRegression', 'DecisionTreeRegressor', 'RandomForestRegressor', 'KNeighborsRegressor', 'GradientBoostingRegressor']

    # Validation of models
    '''
    for d in [5, 8, 10]:
        data_store27_dept = find_department(data, store=27, dept=d, drop_list=drop_list)
        X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test = split_data(data_store27_dept)

        for n in name:
            print(n)

            if n == 'LinearRegression':
                lr = func(X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test, store=27,
                                                        dept=d, models_name=n, plot_test=False, val=True)

            elif n == 'DecisionTreeRegressor':
                for c in ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']:
                    for s in ['best', 'random']:
                        print('criterion:'+c+', splitter:'+s)
                        dtr = func(X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test, store=27,
                                                        dept=d, models_name=n, plot_test=False, val=True, par1=c, par2=s)

            elif n == 'RandomForestRegressor':
                for c in ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']:
                    for e in [50, 100, 200, 300, 400, 500]:
                        print('n_estimators:'+str(e)+', criterion:'+c)
                        rfr = func(X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test, store=27,
                                                        dept=d, models_name=n, plot_test=False, val=True, par1=c, par3=e)

            elif n == 'KNeighborsRegressor':
                for w in ['uniform', 'distance']:
                    for m in ['manhattan', 'euclidean', 'cityblock']:
                        for n_n in range(2, 21):
                            print('n_neighbors:'+str(n_n)+', weights:'+w+', metric:'+m)
                            knr = func(X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test,
                                                        store=27, dept=d, models_name=n, plot_test=False, val=True,
                                                        par1=w, par2=m, par3=n_n)

            elif n == 'GradientBoostingRegressor':
                for l in ['squared_error', 'absolute_error', 'huber', 'quantile']:
                    for e in [50, 100, 200, 300, 400, 500]:
                        for lr in [0.01, 0.05, 0.1, 0.5, 1]:
                            print('loss:'+l+', n_estimators:'+str(e)+', learning_rate:'+str(lr))
                            gbr = func(X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test,
                                                        store=27, dept=d, models_name=n, plot_test=True, val=True,
                                                        par1=l, par3=lr, par4=e)
    '''

    # Test of models
    '''
    d = 10 # 5, 8
    data_store27_dept = find_department(data, store=27, dept=d, drop_list=drop_list)
    X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test = split_data(data_store27_dept)

    for n in name:
        print(n)

        if n == 'LinearRegression':
            lr = func(X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test, store=27,
                      dept=d, models_name=n, plot_test=False, val=False)

        elif n == 'DecisionTreeRegressor':
            c = 'friedman_mse'
            s = 'best'
            print('criterion:' + c + ', splitter:' + s)
            dtr = func(X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test, store=27,
                       dept=d, models_name=n, plot_test=False, val=False, par1=c, par2=s)

        elif n == 'RandomForestRegressor':
            c = 'squared_error'
            e = 50
            print('n_estimators:' + str(e) + ', criterion:' + c)
            rfr = func(X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test, store=27,
                        dept=d, models_name=n, plot_test=False, val=False, par1=c, par3=e)

        elif n == 'KNeighborsRegressor':
            w = 'uniform'
            m = 'manhattan'
            n_n = 13
            print('n_neighbors:' + str(n_n) + ', weights:' + w + ', metric:' + m)
            knr = func(X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test, store=27,
                        dept=d, models_name=n, plot_test=False, val=False, par1=w, par2=m, par3=n_n)

        elif n == 'GradientBoostingRegressor':
            l = 'absolute_error'
            e = 100
            lr = 0.5
            print('loss:' + l + ', n_estimators:' + str(e) + ', learning_rate:' + str(lr))
            gbr = func(X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test, store=27,
                        dept=d, models_name=n, plot_test=True, val=False, par1=l, par3=lr, par4=e)
    '''

    # Best models

    # With MarkDowns
    drop_list = ['Store', 'Dept', 'Type', 'Size']
    print('With MarkDowns')

    # Department 5
    data_store27_dept = find_department(data, store=27, dept=5, drop_list=drop_list)
    X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test = split_data(data_store27_dept)
    print('loss: absolute_error, n_estimators: 50, learning_rate: 0.01')
    gbr = func(X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test, store=27, dept=5,
               models_name='GradientBoostingRegressor', plot_test=False, val=False, par1='absolute_error', par3=0.01, par4=50)

    # Department 8
    data_store27_dept = find_department(data, store=27, dept=8, drop_list=drop_list)
    X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test = split_data(data_store27_dept)
    print('n_neighbors: 20, weights: uniform, metric: euclidean')
    knr = func(X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test, store=27, dept=8,
               models_name='KNeighborsRegressor', plot_test=False, val=False, par1='uniform', par2='euclidean', par3=20)

    # Department 10
    data_store27_dept = find_department(data, store=27, dept=10, drop_list=drop_list)
    X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test = split_data(data_store27_dept)
    print('n_neighbors: 13, weights: uniform, metric: manhattan')
    knr = func(X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test, store=27, dept=10,
               models_name='KNeighborsRegressor', plot_test=False, val=False, par1='uniform', par2='manhattan', par3=13)

    # Without MarkDowns
    drop_list = ['Store', 'Dept', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Type', 'Size']
    print('\nWithout MarkDowns')

    # Department 5
    data_store27_dept = find_department(data, store=27, dept=5, drop_list=drop_list)
    X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test = split_data(data_store27_dept)
    print('loss: absolute_error, n_estimators: 50, learning_rate: 0.01')
    gbr = func(X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test, store=27, dept=5,
               models_name='GradientBoostingRegressor', plot_test=False, val=False, par1='absolute_error', par3=0.01, par4=50)

    # Department 8
    data_store27_dept = find_department(data, store=27, dept=8, drop_list=drop_list)
    X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test = split_data(data_store27_dept)
    print('loss: absolute_error, n_estimators: 400, learning_rate: 0.01')
    gbr = func(X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test, store=27, dept=8,
               models_name='GradientBoostingRegressor', plot_test=False, val=False, par1='absolute_error', par3=0.01, par4=400)

    # Department 10
    data_store27_dept = find_department(data, store=27, dept=10, drop_list=drop_list)
    X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test = split_data(data_store27_dept)
    print('n_neighbors: 13, weights: uniform, metric: manhattan')
    knr = func(X_dept_train, X_dept_val, X_dept_test, y_dept_train, y_dept_val, y_dept_test, store=27, dept=10,
               models_name='KNeighborsRegressor', plot_test=False, val=False, par1='uniform', par2='manhattan', par3=13)
