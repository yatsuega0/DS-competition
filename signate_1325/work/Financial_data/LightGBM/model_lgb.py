import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# warningsを非表示にする
warnings.filterwarnings("ignore")

# %matplotlib inline

y_col = "MIS_Status"

cat_col = [
    "RevLineCr",
    "LowDoc",
    "Sector",
    "State",
    "BankState",
    "FranchiseCode",
]

params_lgb = {
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "colsample_bytree": 0.8,
    "subsample_freq": 1,
    "subsample": 0.8,
    "random_seed": 0,
}


train_data = pd.read_csv("train.csv", index_col=0)
test_data = pd.read_csv("test.csv", index_col=0)
ss = pd.read_csv("sample_submission.csv", header=None)

# train_data.head()

def basic_info(df):
    rows = []
    for col in df.columns:
        rows.append([col, df[col].dtype, df[col].isnull().sum(), len(df[col].unique())])
    return pd.DataFrame(rows, columns=["col", "type", "num_NaN", "val_warm"])


def preprocessing(df, replace_dict=None, ce_dict=None):
    """
    データフレームに対する前処理を行います。

    Parameters:
    - df (pd.DataFrame): 前処理を行うデータフレーム。
    - replace_dict (dict, optional): label encode のための辞書。列名を入れると対応する label encode の数字が得られます。
    - ce_dict (dict, optional): カテゴリカル変数のデータ量を格納する辞書。列名を入れるとそのカテゴリのデータがどのくらいあるかがわかります。

    Returns:
    - pd.DataFrame: 前処理が適用されたデータフレーム。
    - dict: label encode 用の辞書。列名を入れると対応する label encode の数字が得られます。
    - dict: カテゴリカル変数のデータ量を格納する辞書。列名を入れるとそのカテゴリのデータがどのくらいあるかがわかります。
    """
    # Cityは汎用性が低いと考えられるためDrop
    df = df.drop("City", axis=1)

    # Sector, FranchiseCode
    # 32,33→31, 45→44, 49→48に変換
    code_dict = {
        32: 31,
        33: 31,
        45: 44,
        49: 48
    }
    df["Sector"] = df["Sector"].replace(code_dict)

    # RevLineCr, LowDoc
    # YN以外　→ NaN
    revline_dict = {'0': np.nan, 'T': np.nan}
    df["RevLineCr"] = df["RevLineCr"].replace(revline_dict)

    lowdoc_dict = {'C': np.nan, '0': np.nan, 'S': np.nan, 'A': np.nan}
    df["LowDoc"] = df["LowDoc"].replace(lowdoc_dict)

    # DisbursementDate, ApprovalDate
    # 日付型へ変更し年を抽出
    df['DisbursementDate'] = pd.to_datetime(df['DisbursementDate'], format='%d-%b-%y')
    df["DisbursementYear"] = df["DisbursementDate"].dt.year
    df.drop(["DisbursementDate", "ApprovalDate"], axis=1, inplace=True)

    # 本来数値型のものを変換する
    cols = ["DisbursementGross", "GrAppv", "SBA_Appv"]
    df[cols] = df[cols].applymap(lambda x: x.strip().replace('$', '').replace(',', '')).astype(float).astype(int)

    # 特徴量エンジニアリング
    df["FY_Diff"] = df["ApprovalFY"] - df["DisbursementYear"]
    # 変数としてあまり働かない
    # df["State_is_BankState"] = (df["State"] == df["BankState"])
    # df["State_is_BankState"] = df["State_is_BankState"].replace({True: 1, False: 0})

    df['SBA_Portion'] = df['SBA_Appv'] / df['GrAppv']
    df["DisbursementGrossRatio"] = df["DisbursementGross"] / df["GrAppv"]
    df["MonthlyRepayment"] = df["GrAppv"] / df["Term"]
    df["NullCount"] = df.isnull().sum(axis=1)

    # カテゴリカル変数の設定  nanと新規値: -1とする
    df[cat_col] = df[cat_col].fillna(-1)

    # train
    if replace_dict is None:
        # countencode, labelencode
        # ce_dict: 列名を入れるとそのカテゴリのデータがどのくらいあるかを返してくれます
        # replace_dict: 列名を入れるとlabelencodeのための数字を返す
        ce_dict = {}
        replace_dict = {}
        for col in cat_col:
            replace_dict[col] = {}
            vc = df[col].value_counts()
            ce_dict[col] = vc
            replace_dict_in_dict = {}
            for i, k in enumerate(vc.keys()):
                replace_dict_in_dict[k] = i
            replace_dict[col] = replace_dict_in_dict
            df[f"{col}_CountEncode"] = df[col].replace(vc).astype(int)
            df[col] = df[col].replace(replace_dict_in_dict).astype(int)
        return df, replace_dict, ce_dict

    # test
    else:
        for col in cat_col:
            # Count Encode
            test_vals_uniq = df[col].unique()
            ce_dict_in_dict = ce_dict[col]
            for test_val in test_vals_uniq:
                if test_val not in ce_dict_in_dict.keys():
                    ce_dict_in_dict[test_val] = -1
            df[f"{col}_CountEncode"] = df[col].replace(ce_dict_in_dict).astype(int)

            # Label Encode
            test_vals_uniq = df[col].unique()
            replace_dict_in_dict = replace_dict[col]
            for test_val in test_vals_uniq:
                if test_val not in replace_dict_in_dict.keys():
                    replace_dict_in_dict[test_val] = -1
            df[col] = df[col].replace(replace_dict_in_dict).astype(int)
        return df
    

def MIS_Status_corr_confirm(df, y_col):
    """
    MIS_statusとの相関を確認
    
    Returns:
    - pd.DataFrame: 各変数のMIS_stautsとの相関係数を平均値でソート(ピアソンとスピアマンでそれぞれ相関を見る)
    """
    s_per = df.corr("pearson")[y_col].sort_values()
    s_spr = df.corr("spearman")[y_col].sort_values()
    df_corr = pd.concat([s_per, s_spr], axis=1)
    df_corr.columns = ["Pearson", "Spearman"]

    # 平均値でソート
    return df_corr.loc[df_corr.mean(axis=1).sort_values(ascending=False).keys(), :].drop(y_col)


def f1_optimization(val_y, preds_y_proba):
    """
    F1-scoreを最適化するための閾値を見つける関数。

    Parameters:
    - val_y (array-like): 真のラベル値。
    - preds_y_proba (array-like): モデルの予測確率。

    Returns:
    - float: 最大の平均F1スコア。
    - float: 最適な閾値。
    """
    mean_f1_list = []
    fpr, tpr, thresholds = metrics.roc_curve(val_y, preds_y_proba)
    for threshold in thresholds:
        preds_y = [1 if prob > threshold else 0 for prob in preds_y_proba]
        mean_f1_list.append(f1_score(val_y, preds_y, average='macro'))
    return np.max(mean_f1_list), thresholds[np.argmax(mean_f1_list)]


def model_lgb(X_train, y_train, params_lgb, cat_col):
    """
    LightGBMモデルを訓練し、クロスバリデーションでAUCとF1スコアを計算する関数。

    Parameters:
    - X_train (pd.DataFrame): 訓練データの特徴量。
    - y_train (array-like): 訓練データの真のラベル。
    - params_lgb (dict): LightGBMモデルのハイパーパラメータ。
    - cat_col (list): カテゴリカル変数の列名リスト。

    Returns:
    - list: 学習されたLightGBMモデルのリスト。
    - list: クロスバリデーション各foldでのAUCスコアのリスト。
    - list: クロスバリーション各foldでのF1スコアのリスト。
    - list: クロスバリーション各foldでの最適なF1スコアの閾値のリスト。
    """
    list_metrics_auc = []
    list_metrics_f1 = []
    list_cutoff = []
    list_models = []
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    for fold, (trn_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        trn_x = X_train.iloc[trn_idx, :]
        trn_y = y_train[trn_idx]
        val_x = X_train.iloc[val_idx, :]
        val_y = y_train[val_idx]
        model_lgb = lgb.LGBMClassifier(**params_lgb)
        model_lgb.fit(
            trn_x, trn_y,
            eval_set=(val_x, val_y),
            callbacks=[lgb.early_stopping(100, verbose=True)],
            categorical_feature=cat_col,
        )
        list_models.append(model_lgb)
        preds_y_proba = model_lgb.predict_proba(val_x)[:, 1]
        auc = roc_auc_score(val_y, preds_y_proba)
        f1, threshold = f1_optimization(val_y, preds_y_proba)
        list_metrics_auc.append(auc)
        list_metrics_f1.append(f1)
        list_cutoff.append(threshold)
        print(f"Fold: {fold}, AUC: {auc}, f1 score: {f1} Threshold: {threshold}")
    print(np.mean(list_metrics_auc), np.mean(list_metrics_f1), np.median(list_cutoff))
        
    return list_models, list_metrics_auc, list_metrics_f1, list_cutoff


def predict_test(test_data, list_models ,list_cutoff):
    """
    学習済みモデルと閾値を使用して、テストデータに対する予測を行う関数。

    Parameters:
    - test_data (pd.DataFrame): テストデータの特徴量。
    - list_models (list): 学習済みLightGBMモデルのリスト。
    - list_cutoff (list): クロスバリデーション各foldでの最適なF1スコアの閾値のリスト。

    Returns:
    - list: 予測されたラベル（0または1）のリスト。
    """
    threshold = np.median(list_cutoff)
    preds_y_proba = np.zeros(len(test_data))
    for model in list_models:
        preds_y_proba += model.predict_proba(test_data[model.feature_name_])[:, 1] / len(list_models)
    y_preds = [1 if prob > threshold else 0 for prob in preds_y_proba]
    return y_preds


def sumbmit_score(ss, y_pred):
    """
    予測値をコンペ提出用のファイルに代入させてsumbimitファイルをcsvで保存
    
    Parameters:
    - ss (pd.DataFrame): 提出用のサンプルファイル。通常、1列目は識別子やインデックス、2列目は予測値が格納されている。
    - y_pred (array-like): モデルの予測結果。

    Returns:
    - None: 保存が成功した場合は何も返さない。
    """
    ss[1] = y_pred
    ss[1] = ss[1].astype(int)
    print(ss[1].value_counts())
    ss.to_csv("submit_lgb.csv", header=False, index=False)
    

def get_feature_importance(list_models, feature_names):
    """
    訓練されたLightGBMモデルから特徴量の重要度を抽出する関数。

    Parameters:
    - list_models (list): 訓練されたLightGBMモデルのリスト。
    - feature_names (list): 特徴量の名前のリスト。

    Returns:
    - pd.DataFrame: 特徴量の名前と対応する重要度が含まれるデータフレーム。
    """
    feature_importance_df = pd.DataFrame()
    
    for i, model in enumerate(list_models, start=1):
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = feature_names
        fold_importance_df["Importance"] = model.feature_importances_
        fold_importance_df["Fold"] = i
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    # 特徴量ごとにグループ化し、平均重要度を計算
    mean_importance = feature_importance_df.groupby("Feature")["Importance"].mean().reset_index()
    
    # 平均重要度で降順にソート
    mean_importance = mean_importance.sort_values(by="Importance", ascending=False)

    return mean_importance


if __name__ == '__main__' :
    # display(train_data.head())
    # display(basic_info(train_data))
    
    train_data, replace_dict, ce_dict = preprocessing(train_data)
    test_data = preprocessing(test_data, replace_dict=replace_dict, ce_dict=ce_dict)
    # display(test_data.head())
    # display(basic_info(train_data))
    
    # display(MIS_Status_corr_confirm(train_data, y_col))
    
    # X_train = train_data.drop(y_col, axis=1)
    # LowDocとRevLineCr　→ 特徴量としての寄与度が低いのでdropしたら微量ではあるが精度向上
    X_train = train_data.drop(["LowDoc_CountEncode", "RevLineCr_CountEncode", y_col], axis=1)
    y_train = train_data[y_col]
    test_data = test_data.drop(["LowDoc_CountEncode", "RevLineCr_CountEncode"], axis=1)
    
    list_models, list_metrics_auc, list_metrics_f1, list_cutoff = model_lgb(X_train, y_train, params_lgb, cat_col)
    
    y_preds = predict_test(test_data, list_models ,list_cutoff)
    
    sumbmit_score(ss, y_preds)
    
    feature_importance_df = get_feature_importance(list_models, X_train.columns)
    display(feature_importance_df)
    
    # F1-score: 0.6772646
    