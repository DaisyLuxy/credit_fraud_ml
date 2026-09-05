"""
Business cost for thresholds module

Take business cost into consideration when choosing thresholds.
"""

import numpy as np
import matplotlib.pyplot as plt

def find_best_t(model, X_test, y_test, fn_cost=500, fp_cost=5): # Business logic
    proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        y_pred = (proba > t).astype(int)
        fn = ((y_pred == 0) & (y_test == 1)).sum()  # 漏判欺诈
        fp = ((y_pred == 1) & (y_test == 0)).sum()  # 误报
        cost = fn_cost * fn + fp_cost * fp                    
        costs.append(cost)

    best_t = thresholds[np.argmin(costs)]
    # print("Best thresholds:", best_t)
    
    def plot_threshold_cost():
        # 绘制 阈值(x) - 业务成本(y) 曲线
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, costs, color="steelblue", linewidth=2, label="Business Cost")

        # 标注最优阈值点（成本最低）
        plt.scatter(best_t, min(costs), color="crimson", s=80, zorder=5, label="Optimal Threshold")

        # 辅助线 + 标注文字
        plt.axvline(x=best_t, color="crimson", linestyle="--", alpha=0.7)
        plt.title("Threshold vs. Business Cost", fontsize=14)
        plt.xlabel("Classification Threshold", fontsize=12)
        plt.ylabel(f"Total Business Cost ({fn_cost}*FN + {fp_cost}*FP)", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

        clf = model.steps[-1][1]
        auto_name = clf.__class__.__name__
        # 打印最优结果
        print(f"Best Threshold for {auto_name}：{best_t:.4f}")
        print(f"Minimized Cost for {auto_name}：{min(costs)}")
    
    # 挂载绘图函数到主函数，外部可调用
    find_best_t.plot = plot_threshold_cost

    return best_t