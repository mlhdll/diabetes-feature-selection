import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier

def select_features(method, X, y, k=5):
    """
    Belirtilen yöntemi kullanarak en iyi k özelliği seçer.
    Seçilen özellik isimlerinin bir listesini ve varsa puanlarını/sıralamalarını döndürür.
    """
    feature_names = X.columns.tolist()
    selected_features = []
    scores = {}

    if method == 'RFE (Recursive Feature Elimination)':
        # RFE için tahminci olarak Lojistik Regresyon kullanılıyor
        estimator = LogisticRegression(max_iter=1000)
        selector = RFE(estimator, n_features_to_select=k, step=1)
        selector = selector.fit(X, y)
        selected_features = [f for f, s in zip(feature_names, selector.support_) if s]
        # RFE özellikleri seçer. Seçilenler için daha iyi bir "önem" puanı vermek adına,
        # altta yatan eğitilmiş tahmincinin katsaylarına/özellik önemlerine bakabiliriz.
        if hasattr(selector.estimator_, 'coef_'):
            # Lojistik Regresyon (doğrusal modeller) için
            importances = np.abs(selector.estimator_.coef_[0])
        elif hasattr(selector.estimator_, 'feature_importances_'):
            # Ağaç tabanlı modeller için
            importances = selector.estimator_.feature_importances_
        else:
            # Önem metriği yoksa sıralamaya geri dön
            importances = [1.0] * len(selected_features)

        scores = {f: imp for f, imp in zip(selected_features, importances)}

    elif method == 'Mutual Information':
        # mutual_info_classif stokastiktir (rastgele), bu yüzden tekrarlanabilirlik için random_state ayarlıyoruz
        importances = mutual_info_classif(X, y, random_state=42)
        # Kolay sıralama için bir seri oluştur
        feat_importances = pd.Series(importances, index=feature_names)
        selected_features = feat_importances.nlargest(k).index.tolist()
        scores = feat_importances.to_dict()

    elif method == 'LASSO (L1 Regularization)':
        # LASSO, katsayıları sıfıra indirerek özellikleri seçer
        # Alpha parametresini ayarlamamız gerekir, ancak bu demo için standart küçük bir değer seçeceğiz
        # veya sınıflandırma için etkili bir şekilde LASSO olan penalty='l1' ve solver='liblinear' ile LogisticRegression kullanacağız
        lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000, random_state=42)
        lasso.fit(X, y)
        # Sıfır olmayan katsayılara sahip özellikler seçilir
        coefs = np.abs(lasso.coef_[0])
        feat_importances = pd.Series(coefs, index=feature_names)
        selected_features = feat_importances.nlargest(k).index.tolist()
        scores = feat_importances.to_dict()
    
    else:
        # Varsayılan geri dönüş
        return feature_names[:k], {}

    # Kullanıcının daha iyi anlaması için puanları yüzdelere (0-100) normalize et
    # Sadece seçilen özelliklerin puanlarını filtrele
    
    # Önce, puanları sadece seçilen özelliklere göre filtrele (zaten yapılmadıysa)
    selected_scores = {f: scores[f] for f in selected_features if f in scores}
    
    total_score = sum(selected_scores.values())
    if total_score > 0:
        # Puanları yüzde olacak şekilde güncelle
        scores = {f: (v / total_score) * 100 for f, v in selected_scores.items()}
    else:
        scores = selected_scores

    return selected_features, scores
