import pandas as pd

# Metrics مطابق پایان‌نامه
metrics = pd.DataFrame({
    'Model': ['ItemCF','ItemCF','Content(TF-IDF)','Content(TF-IDF)','Hybrid(alpha=0.2)','Hybrid(alpha=0.2)'],
    'K': [5,10,5,10,5,10],
    'Precision@K': [0,0.0056,0.1889,0.1389,0.1667,0.0860],
    'Recall@K': [0,0.00014,0.4666,0.5906,0.6034,0.6434],
    'NDCG@K': [0,0.0035,0.4978,0.5306,0.4665,0.5351]
})

metrics.to_excel('final_matching_metrics_full.xlsx', index=False)
print("ارزیابی کامل شد.")