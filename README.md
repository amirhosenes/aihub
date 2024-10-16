import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# داده‌های نمونه
data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1],
    'label': [0, 1, 0, 1, 0]
}

# تبدیل داده‌ها به DataFrame
df = pd.DataFrame(data)

# تقسیم داده‌ها به ورودی و خروجی
X = df[['feature1', 'feature2']]
y = df['label']

# تقسیم داده‌ها به مجموعه‌های آموزشی و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ایجاد مدل
model = LogisticRegression()

# آموزش مدل
model.fit(X_train, y_train)

# پیش‌بینی با مدل
y_pred = model.predict(X_test)

# محاسبه دقت مدل
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
