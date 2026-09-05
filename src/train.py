"""
训练入口脚本。

用法：
    conda activate fin-risk
    cd project_root
    python -m src.train

Will do：
    1. Load Data
    2. Split training/testing set 
    3. Train 3 models: Logistic Regression baseline, Random Forest, and XGBoost
    4. Evaluate and save 3 models
"""
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd

from src.data.load_data import (
    load_creditcard,
    split_features_and_target,
    stratified_split,
)
from src.utils.metrics import evaluate
from src.utils.threshold import find_best_t

def train_lr(X_train, y_train):
    # 搭建流水线：标准化 + 逻辑回归
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    pipe.fit(X_train, y_train)
    # 评估模型
    # model_evaluation(pipe, X_test, y_test)
    return pipe

def train_rf(X_train, y_train):
    # 树模型不需要标准化，直接训练
    pipe = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=100,       # 决策树数量，可自行修改
            class_weight="balanced",# 适配不平衡样本
            random_state=42
        ))
    ])
    pipe.fit(X_train, y_train)
    # model_evaluation(pipe, X_test, y_test)
    return pipe

def train_xgb(X_train, y_train):
    pipe = Pipeline([
        ("clf", XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,      # 学习率，可自行调参
            scale_pos_weight=10,     # 不平衡样本权重，根据正负比例修改
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        ))
    ])
    pipe.fit(X_train, y_train)
    # model_evaluation(pipe, X_test, y_test)
    return pipe

# def build_pipeline() -> Pipeline:
#     """构建包含标准化和逻辑回归的训练管线。

#     用 Pipeline 而不是分开做，是为了：
#       1. 避免数据泄漏：scaler 只在训练集上 fit
#       2. 训练和推理共用同一套预处理逻辑
#       3. 整体保存和加载模型时不会漏掉预处理器
#     """
#     return Pipeline([
#         ("scaler", StandardScaler()),
#         ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
#     ])


def main():
    """Main training"""
    # 1. Load Data
    print("[1/4] Load Data...")
    df = load_creditcard()
    print(f"      Data shap: {df.shape}")
    print(f"      Proportion of fraudulent samples: {df['Class'].mean():.4%}")

    df = df.sample(n=50000, random_state=42)
    print(f"      Data sample: {df.shape[0]}")


    # 2. Split training/testing set
    print("[2/4] Split training/testing set ...")
    X, y = split_features_and_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, 42)
    print(f"      Training Set: {X_train.shape}, Testing Set: {X_test.shape}")

    # 3. Train the models
    print("[3/4] Train the models...")
    pipe_lr = train_lr(X_train, y_train)
    pipe_rf = train_rf(X_train, y_train)
    pipe_xgb = train_xgb(X_train, y_train)

    # 4. Evaluate and Save models
    print("[4/4] Evaluate and Save models...")
    # y_proba = pipe.predict_proba(X_test)[:, 1]
    # metrics = evaluate_binary(y_test, y_proba)
    # print_metrics(metrics)

    # 批量存储结果
    results = []

    # 录入指标
    results.append(evaluate("Logistic Regression", pipe_lr, X_test, y_test,find_best_t(pipe_lr, X_test, y_test)))
    find_best_t.plot()
    results.append(evaluate("Random Forest", pipe_rf, X_test, y_test,find_best_t(pipe_rf, X_test, y_test)))
    find_best_t.plot()
    results.append(evaluate("XGBoost", pipe_xgb, X_test, y_test,find_best_t(pipe_xgb, X_test, y_test)))
    find_best_t.plot()

    print("")

    # 生成完整对比表格
    result_df = pd.DataFrame(results)
    print("===== Comparison between 3 models =====")
    print(result_df)


    # 保存模型到 models/ 目录，加时间戳避免覆盖
    from datetime import datetime
    timestamp_lr = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path_lr = Path("models") / f"baseline_lr_{timestamp_lr}.joblib"
    model_path_lr.parent.mkdir(exist_ok=True)
    joblib.dump(pipe_lr, model_path_lr)
    print(f"\nLogistic Regression Model has been saved in: {model_path_lr}")
    
    timestamp_rf = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path_rf = Path("models") / f"baseline_rf_{timestamp_rf}.joblib"
    model_path_rf.parent.mkdir(exist_ok=True)
    joblib.dump(pipe_rf, model_path_rf)
    print(f"\nRandom Forest Model has been saved in: {model_path_rf}")
       
    timestamp_xgb = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path_xgb = Path("models") / f"baseline_xgb_{timestamp_xgb}.joblib"
    model_path_xgb.parent.mkdir(exist_ok=True)
    joblib.dump(pipe_xgb, model_path_xgb)
    print(f"\nXGBoost Model has been saved in: {model_path_xgb}")
    


if __name__ == "__main__":
    main()
