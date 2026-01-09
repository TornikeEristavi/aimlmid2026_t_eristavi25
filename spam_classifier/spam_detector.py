import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Silence warnings for a clean report
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 1. Load the data
df = pd.read_csv('t_eristavi25_92358.csv')
X = df[['words', 'links', 'capital_words', 'spam_word_count']]
y = df['is_spam']

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# 3. Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Results
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 5. Prediction Function (Fixed for warnings)
def evaluate_custom_email(text):
    words = len(text.split())
    links = text.count('http') + text.count('www')
    capitals = sum(1 for c in text if c.isupper())
    spam_keywords = ['win', 'prize', 'gift', 'claim', 'urgent', 'free', 'money']
    spam_count = sum(1 for word in text.lower().split() if word in spam_keywords)
    
    # Put into DataFrame to avoid Feature Name warnings
    features = pd.DataFrame([[words, links, capitals, spam_count]], 
                            columns=['words', 'links', 'capital_words', 'spam_word_count'])
    prediction = model.predict(features)
    return "SPAM" if prediction[0] == 1 else "LEGITIMATE"

# 6. Manual Tests (Aggressive spam to beat the -9.29 intercept)
print("\n--- Manual Tests ---")
# To get SPAM, we need many links and capitals
spam_text = "WINNER! CLAIM YOUR FREE PRIZE NOW AT http://win.com http://prize.com http://gift.com http://cash.com CLICK WWW.LINK.COM"
legit_text = "Hey, are you coming to the office today? Let me know."

print(f"Test 1 (Spam): {evaluate_custom_email(spam_text)}")
print(f"Test 2 (Legit): {evaluate_custom_email(legit_text)}")

# 7. Visualizations (Fixed for warnings)
plt.figure(figsize=(6,4))
sns.countplot(x='is_spam', data=df, hue='is_spam', palette='viridis', legend=False)
plt.title('Class Distribution')
plt.savefig('distribution.png')

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Heatmap')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('heatmap.png')
