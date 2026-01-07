import pandas as pd
from src.text_cleaning import clean_text

def prepare_complaints_df(
    df: pd.DataFrame,
    text_col: str = "narrative",
    label_col: str = "Product",
    date_col: str = "Date received",
    min_chars: int = 20,
) -> pd.DataFrame:
    """
    Basic cleaning + filtering for Consumer Complaints dataset.
    - Drops missing text/label/date
    - Parses date
    - Cleans text
    - Drops too-short cleaned text
    """
    df = df[[text_col, label_col, date_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df = df.dropna(subset=[text_col, label_col, date_col])
    df["text_clean"] = df[text_col].map(clean_text)

    df = df[df["text_clean"].str.len() >= min_chars].copy()
    return df


def stratified_cap_per_class(
    df: pd.DataFrame,
    label_col: str,
    cap_per_class: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Samples up to `cap_per_class` rows per label, reproducibly.
    """
    return (
        df.groupby(label_col, group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), cap_per_class), random_state=random_state))
          .copy()
    )


def time_split(
    df: pd.DataFrame,
    date_col: str,
    train_frac: float = 0.8,
):
    """
    Time-based split.
    """
    df = df.sort_values(date_col)
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
