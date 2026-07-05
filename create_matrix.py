import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

# بارگذاری
data = pd.read_csv('synthetic_interactions_final.csv')

# ساخت ماتریس sparse
user_ids = data['user_synth'].astype('category').cat.codes
item_ids = data['product_id_num'].astype('category').cat.codes
ratings = data['rate_num']

interaction_matrix = csr_matrix((ratings, (user_ids, item_ids)))

# ذخیره npz
np.savez('interaction_matrix_train.npz', 
         data=interaction_matrix.data,
         indices=interaction_matrix.indices,
         indptr=interaction_matrix.indptr,
         shape=interaction_matrix.shape)

print("ماتریس تعامل ساخته شد.")