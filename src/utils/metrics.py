"""
Model evaluation module

Provide commonly used evaluation functions for imbalanced binary classification problems.
"""
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score, 
    recall_score, 
    f1_score
)

# 终极升级版：包含全部核心指标（适配不平衡数据，取正类1的指标）
def evaluate(name, model, X_test, y_test, best_t):
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = proba > best_t

    return {
        "model": name,
        "ROC-AUC": round(roc_auc_score(y_test, proba), 4),
        "PR-AUC": round(average_precision_score(y_test, proba), 4),
        # 只统计少数类/正类=1的精准率、召回率、F1（不平衡数据核心指标）
        "Precision": round(precision_score(y_test, y_pred, pos_label=1), 4),
        "Recall": round(recall_score(y_test, y_pred, pos_label=1), 4),
        "F1-Score": round(f1_score(y_test, y_pred, pos_label=1), 4)
    }

# def evaluate_binary(y_true, y_proba, threshold: float = 0.5) -> dict:
#     """计算二分类关键指标。

#     参数：
#         y_true: 真实标签，形如 [0, 1, 0, 0, 1]
#         y_proba: 预测为正类（欺诈）的概率，形如 [0.02, 0.9, 0.1, ...]
#         threshold: 概率转为 0/1 标签的阈值，默认 0.5

#     返回：
#         dict，包含关键指标和分类报告字符串。
#     """
#     # 中文注释：把概率转为 0/1 预测
#     y_pred = (y_proba >= threshold).astype(int)

#     # 中文注释：计算三个核心指标
#     #   - roc_auc: 整体区分能力，样本平衡时看它
#     #   - pr_auc:  少数类识别能力，不平衡时看它（本项目重点）
#     #   - report:  Precision/Recall/F1 分类明细
#     metrics = {
#         "roc_auc": roc_auc_score(y_true, y_proba),
#         "pr_auc": average_precision_score(y_true, y_proba),
#         "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
#         "classification_report": classification_report(
#             y_true, y_pred, digits=4
#         ),
#     }
#     return metrics


# def print_metrics(metrics: dict) -> None:
    
#     print(f"ROC-AUC : {metrics['roc_auc']:.4f}")
#     print(f"PR-AUC  : {metrics['pr_auc']:.4f}")
#     print("\nConfusion Matrix (row=Real, col=prediction):")
#     print(f"  TN={metrics['confusion_matrix'][0][0]:>6}  "
#           f"FP={metrics['confusion_matrix'][0][1]:>6}")
#     print(f"  FN={metrics['confusion_matrix'][1][0]:>6}  "
#           f"TP={metrics['confusion_matrix'][1][1]:>6}")
#     print("\nClassification Report:")
#     print(metrics["classification_report"])
