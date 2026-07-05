import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# بارگذاری
data = pd.read_csv('synthetic_interactions_final.csv')

# ساخت ماتریس کاربر-آیتم
user_item_matrix = csr_matrix((data['rate_num'], 
                               (data['user_synth'].astype('category').cat.codes, 
                                data['product_id_num'].astype('category').cat.codes)))

# محاسبه شباهت آیتم‌ها
item_similarity = cosine_similarity(user_item_matrix.T)

print("ماتریس شباهت Item-CF ساخته شد.")

# تابع توصیه
def recommend_item_cf(user_id, top_k=5):
    user_ratings = user_item_matrix[user_id].toarray().flatten()
    similar_items = item_similarity.dot(user_ratings)
    recommended = np.argsort(similar_items)[::-1][:top_k]
    return recommended

print("Item-CF آماده است.")