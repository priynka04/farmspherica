import pandas as pd

def handle_missing_values(df):
    """Fill missing values using the column mean."""
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    print("✅ Missing values filled with column means")
    return df

def normalize_columns(df):
    """Scale all numeric columns to 0–1 range."""
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max != col_min:
            df[f'{col}_normalized'] = (df[col] - col_min) / (col_max - col_min)
    print("✅ Normalization complete")
    return df

def add_rolling_averages(df, date_col='date', window=7):
    """Add 7-day rolling average for each numeric column."""
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        df[f'{col}_7day_avg'] = df[col].rolling(window=window, min_periods=1).mean()
    print(f"✅ {window}-day rolling averages added")
    return df

def add_rate_of_change(df):
    """Calculate how much each parameter changes row to row."""
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        df[f'{col}_change'] = df[col].diff()
    print("✅ Rate of change columns added")
    return df

def export_clean_csv(df, output_path="data/clean_dataset.csv"):
    """Save the preprocessed data to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"✅ Clean dataset saved to {output_path}")

def run_preprocessing(input_path="data/sample_data.csv"):
    df = pd.read_csv(input_path)
    df = handle_missing_values(df)
    df = normalize_columns(df)
    df = add_rolling_averages(df)
    df = add_rate_of_change(df)
    export_clean_csv(df)
    return df

if __name__ == "__main__":
    run_preprocessing()
    
