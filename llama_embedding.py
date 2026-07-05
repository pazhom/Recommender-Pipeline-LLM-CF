from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# بارگذاری مدل LLaMA (نسخه سبک یا کامل)
model_name = "meta-llama/Llama-2-7b-hf"  # یا نسخه کوچکتر اگر RAM کم است
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# تابع استخراج embedding
def get_embedding(text, max_length=256):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # میانگین pooling
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# بارگذاری داده
reviews = pd.read_excel('Book1.xlsx')
reviews['body_clean'] = reviews['body'].astype(str).apply(lambda x: x.strip())

# استخراج embedding
embeddings = []
for text in tqdm(reviews['body_clean']):
    emb = get_embedding(text)
    embeddings.append(emb)

embeddings = np.array(embeddings)
np.save('llama_embeddings.npy', embeddings)

print("Embeddingهای LLaMA ذخیره شد در llama_embeddings.npy")