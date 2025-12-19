from pathlib import Path
import pandas as pd

DATA_PATH_DEFAULT = Path("data/raw/subsampled_data.csv")

def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def report_basic_stats(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nHead:\n", df.head())
    print("\nMissing values per column:\n", df.isna().sum().sort_values(ascending=False).head(20))
    print("\nDescribe (first 20 rows):\n", df.describe(include="all").T.head(20))

def main():
    df = load_dataset(DATA_PATH_DEFAULT)
    report_basic_stats(df)

if __name__ == "__main__":
    main()
