"""
Data Load and split module

Load credit card spam data from original csv and split for training and test.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# 默认数据路径，用 Path 对象比字符串拼接更安全跨平台
DEFAULT_DATA_PATH = Path("data/raw/creditcard.csv")


def load_creditcard(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load credit card spam data from original csv

    parameter：
        path: CSV data path, from project root, data/raw/creditcard.csv

    return：
        pd.DataFrame，include all raw fields
    """
    # 转 Path 便于统一处理
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not exist: {path.resolve()}\n"
            "Please confirm already downloaded creditcard.csv from Kaggle and put in path data/raw/ "
        )
    return pd.read_csv(path)


def split_features_and_target(
    df: pd.DataFrame,
    target_col: str = "Class",
) -> tuple[pd.DataFrame, pd.Series]:

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """stratify train and test set, keep the positive/negative ratio the same

    parameter：
        X: feature
        y: tag
        test_size
        random_state

    return：
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,       
        random_state=random_state,
    )
