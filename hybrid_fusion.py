import numpy as np

# بارگذاری embedding و CF scores (نمونه)
# llm_scores = np.load('llama_scores.npy')
# cf_scores = np.load('cf_scores.npy')

# Fusion
def hybrid_score(cf_score, llm_score, alpha=0.2):
    return alpha * cf_score + (1 - alpha) * llm_score

# مثال
cf = np.array([0.1, 0.3, 0.8])
llm = np.array([0.7, 0.6, 0.4])
final_scores = hybrid_score(cf, llm, alpha=0.2)

print("امتیاز نهایی Hybrid:", final_scores)
print("Hybrid Fusion با α=0.2 آماده است.")