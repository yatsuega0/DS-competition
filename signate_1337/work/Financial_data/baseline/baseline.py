import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix, make_scorer
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
sample_submit = pd.read_csv('sample_submission.csv', index_col=0, header=None)
print(train_data.shape)
train_data.head()

X = train_data.drop(["Unnamed: 0", "MIS_Status"], axis=1)
y = train_data["MIS_Status"]
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].astype("category")
        
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
}

#Baseline
def s1_score(y_true, y_pred):
    # 混同行列の取得
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # s1-scoreの計算
    sensitivity = tp / (tp + fn)  # Recall
    precision = tp / (tp + fp)    # Precision

    s1 = 2 * (sensitivity * precision) / (sensitivity + precision) if (sensitivity + precision) > 0 else 0

    return s1


def baseline(X, y, params, cv, s1_scorer):
    
    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # LightGBMモデルの学習
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        # テストデータで予測
        y_pred = model.predict(X_test)

        # s1-scoreの計算
        s1 = s1_score(y_test, y_pred)
        print(f"Fold {i+1} - s1-score: {s1:.4f}")

        # 特徴量の重要度を取得
        feature_importance = model.feature_importances_

        # 特徴量の名前を取得
        feature_names = X.columns

        # 重要度をデータフレームに格納
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

        # 重要度で降順にソート
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # バー プロットで可視化
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance Plot - Fold {i+1}')
        plt.show()
        
    return y_pred, feature_importance_df


def test_prediction(test_data):
    X_test = test_data.copy()
    X_test = test_data.drop("Unnamed: 0", axis=1)
    for col in X_test.columns:
        if X_test[col].dtype == "object":
            X_test[col] = X_test[col].astype("category")
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    predictions = model.predict(X_test)
    sample_submit = pd.read_csv('sample_submission.csv', index_col=0, header=None)
    sample_submit[1] = predictions
    # display(sample_submit)
    sample_submit.to_csv('submit_baseline.csv', header=None)


if __name__ == '__main__' :
    s1_scorer = make_scorer(s1_score, greater_is_better=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred, feature_importance_df = baseline(X, y, params, cv, s1_scorer)
    test_prediction(test_data)
    # F1-score 0.6288603
    