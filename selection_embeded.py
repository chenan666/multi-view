import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy as sc
from xgboost import XGBRegressor as xgbr

inputfile = r"data_path"

def main():
    df = pd.read_csv(inputfile, encoding='utf-8')
    df.drop(['formula'], axis=1, inplace=True)
    feature_names = df.drop(['target'], axis=1).columns.tolist()
    features = np.array(df.drop(['target'], axis=1))
    target = np.array(df['target'])
    training_features, testing_features, training_target, testing_target = \
        train_test_split(features, target, test_size=0.2, random_state=0)

    exported_pipeline = xgbr(base_score=0.3, booster='gbtree', colsample_bylevel=1,
                             colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
                             max_depth=4, min_child_weight=0.5, n_estimators=190,
                             n_jobs=1, random_state=0,
                             reg_alpha=0, reg_lambda=2, scale_pos_weight=1, seed=None,
                             subsample=0.6)
    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)
    print(results)
    accuracy = exported_pipeline.score(testing_features, testing_target)
    print("accuracy: %s" % accuracy)
    trainingaccuracy = exported_pipeline.score(training_features, training_target)
    print("trainingaccuracy: %s" % trainingaccuracy)
    print('r2_score(越接近1越好)：', r2_score(testing_target, exported_pipeline.predict(testing_features)))
    mse = mean_squared_error(testing_target, exported_pipeline.predict(testing_features))
    print("testingMSE: %.4f" % mse)
    mse2 = mean_squared_error(training_target, exported_pipeline.predict(training_features))
    print("trainingMSE: %.4f" % mse2)
    r1 = sc.stats.pearsonr(testing_target, exported_pipeline.predict(testing_features))
    print("testingr:", r1)
    r2 = sc.stats.pearsonr(training_target, exported_pipeline.predict(training_features))
    print("trainingr:", r2)

    # Plot feature importance
    feature_importance = exported_pipeline.feature_importances_
    # Make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    # Combine feature names and importance scores
    features_with_importance = list(zip(feature_names, feature_importance))

    # Remove features with 'Data' in the name and importance == 0
    filtered_features_with_importance = [
        (name, importance) for name, importance in features_with_importance
        if 'Data' not in name and importance != 0
    ]

    # Create a new DataFrame without the filtered features
    filtered_feature_names = [name for name, _ in filtered_features_with_importance]
    filtered_df = df[filtered_feature_names + ['target']]

    # Save the new DataFrame to a CSV file
    outputfile = r"saved_data_path"
    filtered_df.to_csv(outputfile, index=False)

    print(f"New DataFrame saved to {outputfile}")

    # Sort by importance score
    filtered_features_with_importance.sort(key=lambda x: x[1])

    # Get the 10 least important features
    least_important_features = filtered_features_with_importance[:10]

    # Print the 10 least important features
    print("the 10 least important features:")
    for feature, importance in least_important_features:
        print(f"feature: {feature}, importance: {importance:.2f}%")

if __name__ == "__main__":
    main()
