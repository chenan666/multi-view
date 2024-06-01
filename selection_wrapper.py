import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor as xgbr
from sklearn.feature_selection import RFE

inputfile4 = r"data_path"

def main():
    df = pd.read_csv(inputfile4, encoding='utf-8')

    df.drop(['formula'], axis=1, inplace=True)
    features = np.array(df.drop(['target'], axis=1))
    target = np.array(df['target'])
    training_features, testing_features, training_target, testing_target = \
                train_test_split(features, target, test_size=0.2, random_state=0)

    # model
    model = xgbr(base_score=0.3, booster='gbtree', colsample_bylevel=1,
                 colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
                 max_depth=4, min_child_weight=0.5, n_estimators=190,
                 n_jobs=1, random_state=0, reg_alpha=0, reg_lambda=2,
                 scale_pos_weight=1, subsample=0.6)

    # RFE
    rfe = RFE(model, n_features_to_select=int(features.shape[1] / 2), step=1)
    rfe = rfe.fit(training_features, training_target)

    # Get features
    training_features_rfe = rfe.transform(training_features)
    testing_features_rfe = rfe.transform(testing_features)

    # Training
    model.fit(training_features_rfe, training_target)
    results = model.predict(testing_features_rfe)
    crossScore = cross_val_score(model, training_features_rfe, training_target, cv=10)

    print('results：')
    print(results)

    print('Score')
    print(model.score(testing_features_rfe, testing_target))
    print('crossScore:')
    print(crossScore)
    print(crossScore.mean())
    mse2 = mean_squared_error(training_target, model.predict(training_features_rfe))
    print("trainingMSE: %.4f" % mse2)
    mse = mean_squared_error(testing_target, model.predict(testing_features_rfe))
    print("testingMSE: %.4f" % mse)
    rmse = np.sqrt(mean_squared_error(testing_target, model.predict(testing_features_rfe)))
    print("testingRMSE: %.4f" % rmse)

if __name__ == "__main__":
    main()
