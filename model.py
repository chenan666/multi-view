#xgbr
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import joblib
from xgboost.sklearn import XGBRegressor as xgbr

from sklearn.model_selection import cross_val_score
from scipy.optimize import curve_fit



import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')

inputfile = r"data_path"


def main():
    df = pd.read_csv(inputfile, encoding='utf-8')

    df.drop(['formula'],axis=1,inplace=True)
    features = np.array(df.drop(['target'],axis=1))
    target = np.array(df['target'])
    training_features, testing_features, training_target, testing_target = \
                train_test_split(features, target, test_size=0.2,random_state=0)

    exported_pipeline = xgbr(base_score=0.3, booster='gbtree', colsample_bylevel=1,
                             colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
                             max_depth=4, min_child_weight=0.5,  n_estimators=190,
                             n_jobs=1, nthreaad=None, random_state=0,
                             reg_alpha=0, reg_lambda=2, scale_pos_weight=1, seed=None,
                             subsample=0.6)

    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)
    crossScore= cross_val_score(exported_pipeline, training_features,training_target, cv=10)

    print('results：')
    print(results)

    print('Score')
    print(exported_pipeline.score(testing_features, testing_target))
    print('crossScore:')
    print(crossScore)
    print(crossScore.mean())
    mse2 = mean_squared_error(training_target, exported_pipeline.predict(training_features))
    print("trainingMSE: %.4f" % mse2)
    mse = mean_squared_error(testing_target, exported_pipeline.predict(testing_features))
    print("testingMSE: %.4f" % mse)
    rmse = np.sqrt(mean_squared_error(testing_target, exported_pipeline.predict(testing_features)))
    print("testingRMSE: %.4f" % rmse)

# Plot training deviance
#Save model
    #joblib.dump(exported_pipeline,'xgbrver604.pkl')

#Plot the difference between testing and training target
    def func(x,a,b):
        return a+b*x
    popt, pcov = curve_fit(func, training_target, exported_pipeline.predict(training_features))

    plt.subplot(1,1,1)
    plt.title('difference')
    plt.plot(training_target, exported_pipeline.predict(training_features), 'o', color='tomato', alpha= 0.5, label='Training Set')
    plt.plot(testing_target, exported_pipeline.predict(testing_features), 'o', color='royalblue', alpha= 0.5, label='Testing Set')
    # print(training_target)
    # print(exported_pipeline.predict(training_features))
    # print(testing_target)
    # print(exported_pipeline.predict(testing_features))

    # plot data
    training_data = {
        "training_target": training_target,
        "predicted_training": exported_pipeline.predict(training_features)
    }
    df_training = pd.DataFrame(training_data)

    testing_data = {
        "testing_target": testing_target,
        "predicted_testing": exported_pipeline.predict(testing_features)
    }
    df_testing = pd.DataFrame(testing_data)

    # saved path
    training_csv_file_path = r"saved_training_data_path"
    testing_csv_file_path = r"saved_testing_data_path"

    # save
    df_training.to_csv(training_csv_file_path, index=False)
    df_testing.to_csv(testing_csv_file_path, index=False)


    plt.legend(loc='best')
    plt.xlabel('DFT Calculations')
    plt.ylabel('XGBoostRegression')
    popt, pcov = curve_fit(func, training_target, exported_pipeline.predict(training_features))
    popt1, pcov1 = curve_fit(func, testing_target, exported_pipeline.predict(testing_features))
    yy2 = [func(i,popt[0], popt[1]) for i in training_target]
    yy3 = [func(i,popt[0], popt[1]) for i in testing_target]
    plt.plot(training_target,yy2,'c-',ls='-',color='cornflowerblue')
    plt.plot()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
