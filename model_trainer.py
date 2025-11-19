from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

def train_evaluate(model_type, X, y, test_size=0.2, random_state=42):
    """
    Seçilen modeli eğitir ve değerlendirir.
    Metrikleri ve eğitilmiş modeli döndürür.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if model_type == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Bilinmeyen model türü: {model_type}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm,
        'model': model
    }
