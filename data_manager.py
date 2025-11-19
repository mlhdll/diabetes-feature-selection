import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath='diabetes.csv'):
    """
    Diyabet veri setini yükler ve temel ön işlemeyi gerçekleştirir.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return None, "Dosya bulunamadı."

    # 0 değerlerinin geçersiz olduğu sütunlarda 0'ı NaN ile değiştir
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

    # NaN değerlerini ortalama ile doldur (bu proje için basit strateji)
    df.fillna(df.mean(), inplace=True)

    return df, None

def preprocess_data(df, target_column='Outcome'):
    """
    Özellikleri ve hedef değişkeni ayırır, özellikleri ölçeklendirir.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y
