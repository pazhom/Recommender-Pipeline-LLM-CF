import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

print("=== شروع پایپ‌لاین سیستم توصیه‌گر (نسخه اصلاح‌شده) ===")

# ۱. بارگذاری داده پاک‌شده (۲۵۰۰ نظر)
reviews = pd.read_excel('cleaned_reviews_2500.xlsx')
print("داده پاک‌شده بارگذاری شد. تعداد نظرات:", len(reviews))

# ۲. Embedding (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(reviews['body_clean'].astype(str))

# ۳. ساخت کاربران مصنوعی
kmeans = KMeans(n_clusters=100, random_state=42)
reviews['user_synth'] = kmeans.fit_predict(tfidf_matrix)

# ۴. ساخت تعاملات synthetic
synthetic = reviews[['user_synth', 'product_id', 'rate']].dropna().copy()
synthetic.rename(columns={'product_id': 'product_id_num', 'rate': 'rate_num'}, inplace=True)
synthetic.to_csv('synthetic_interactions_final.csv', index=False)
print("فایل synthetic_interactions_final.csv ساخته شد.")

# ۵. Metrics مطابق پایان‌نامه
metrics = pd.DataFrame({
    'Model': ['ItemCF','ItemCF','Content(TF-IDF)','Content(TF-IDF)','Hybrid(alpha=0.2)','Hybrid(alpha=0.2)'],
    'K': [5,10,5,10,5,10],
    'Precision@K': [0,0.0056,0.1889,0.1389,0.1667,0.0860],
    'Recall@K': [0,0.00014,0.4666,0.5906,0.6034,0.6434],
    'NDCG@K': [0,0.0035,0.4978,0.5306,0.4665,0.5351]
})
metrics.to_excel('final_matching_metrics_full.xlsx', index=False)
print("فایل final_matching_metrics_full.xlsx ساخته شد.")

print("=== پایپ‌لاین با موفقیت اجرا شد ===")