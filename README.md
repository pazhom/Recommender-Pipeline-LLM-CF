# پایپ‌لاین سیستم توصیه‌گر ترکیبی LLM + CF

## معرفی پروژه
این ریپازیتوری شامل **پایپ‌لاین کامل** پیاده‌سازی سیستم توصیه‌گر ترکیبی مبتنی بر مدل زبانی بزرگ (LLaMA) و فیلترینگ مشارکتی (CF) است. پروژه از داده خام شروع شده و تا تولید خروجی نهایی (معیارهای ارزیابی) ادامه دارد.

## ساختار پروژه
- `preprocess.py` → پیش‌پردازش و پاک‌سازی داده خام (از ۳۰۰۰ به ۲۵۰۰ نظر)
- `llama_embedding.py` → استخراج Embedding معنایی با LLaMA
- `item_cf.py` → پیاده‌سازی Item-Based Collaborative Filtering
- `hybrid_fusion.py` → ادغام Hybrid (α=0.2)
- `create_matrix.py` → ساخت ماتریس تعامل sparse و فایل `.npz`
- `evaluate.py` → محاسبه معیارهای Precision، Recall، NDCG
- `recommender_pipeline.py` → فایل اصلی که همه مراحل را اجرا می‌کند

## فایل‌های خروجی
- `cleaned_reviews_2500.xlsx` → داده پاک‌سازی‌شده
- `synthetic_interactions_final.csv` → تعاملات مصنوعی
- `final_matching_metrics_full.xlsx` → نتایج ارزیابی (مطابق پایان‌نامه)
- `interaction_matrix_train.npz` → ماتریس تعامل آموزش

## نحوه اجرا
1. فایل `preprocess.py` را اجرا کنید.
2. فایل `recommender_pipeline.py` را اجرا کنید.
3. خروجی‌ها در پوشه ایجاد می‌شوند.

## نتایج کلیدی
- Hybrid (α=0.2): Precision@5 = 0.1667، NDCG@10 = 0.5351
- بهبود قابل توجه در سناریوهای Cold-Start و Sparsity

## وابستگی‌ها
```bash
pip install pandas numpy scikit-learn transformers torch tqdm openpyxl
