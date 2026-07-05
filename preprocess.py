import pandas as pd

print("=== پیش‌پردازش داده خام ===")

# بارگذاری داده خام
reviews = pd.read_excel('Book1.xlsx')
print("تعداد نظرات اولیه:", len(reviews))

# پاک‌سازی مطابق پایان‌نامه
reviews_clean = reviews.dropna(subset=['body', 'rate'])
reviews_clean = reviews_clean[reviews_clean['rate'].between(1, 5)]  # نرخ معتبر
reviews_clean = reviews_clean.drop_duplicates(subset=['body'])     # حذف تکراری

# محدود به ۲۵۰۰ نظر (دقیق مطابق پایان‌نامه)
reviews_clean = reviews_clean.head(2500)

reviews_clean['body_clean'] = reviews_clean['body'].astype(str).apply(lambda x: x.strip().lower())

reviews_clean.to_excel('cleaned_reviews_2500.xlsx', index=False)
print("تعداد نظرات بعد از پاک‌سازی:", len(reviews_clean))
print("فایل cleaned_reviews_2500.xlsx ساخته شد.")