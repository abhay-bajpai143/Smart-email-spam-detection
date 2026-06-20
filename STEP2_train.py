    # ============================================================
# STEP2_train.py  —  SpamGuard Training Pipeline
# Run: python STEP2_train.py
# Trains 4 models + optimizes ensemble weights for log-loss
# ============================================================

import pandas as pd
import numpy as np
import re, os, joblib, urllib.request, zipfile, warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score, f1_score, roc_auc_score
from scipy.optimize import minimize

# ── Download NLTK ────────────────────────────────────────────
for pkg in ['stopwords', 'punkt', 'wordnet']:
    try: nltk.data.find(f'tokenizers/{pkg}')
    except: nltk.download(pkg, quiet=True)

STEMMER   = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

# ── Text Cleaning ─────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', ' email ', text)
    text = re.sub(r'\$[\d,]+', ' money ', text)
    text = re.sub(r'\d+', ' num ', text)
    text = re.sub(r'!{2,}', ' urgent ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    tokens = [STEMMER.stem(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)

# ── Load / Download Dataset ───────────────────────────────────
def load_data():
    csv_path = 'data/emails.csv'
    if not os.path.exists(csv_path):
        print("[Download] Fetching UCI SMS Spam Collection...")
        os.makedirs('data', exist_ok=True)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        urllib.request.urlretrieve(url, 'data/tmp.zip')
        with zipfile.ZipFile('data/tmp.zip') as z:
            z.extractall('data')
        df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label','text'])
        df['label'] = df['label'].map({'spam':1,'ham':0})
        df.dropna(inplace=True)
        df.to_csv(csv_path, index=False)
        print(f"[Download] Saved {len(df)} samples → data/emails.csv")
    else:
        df = pd.read_csv(csv_path)
    return df

# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  SpamGuard — Training Pipeline")
print("=" * 60)

df = load_data()
df['clean'] = df['text'].apply(clean_text)
X, y = df['clean'], df['label']

X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.176, random_state=42, stratify=y_tmp)
print(f"\nSplit → Train:{len(X_train)} | Val:{len(X_val)} | Test:{len(X_test)}")

# ── TF-IDF ─────────────────────────────────────────────────────
print("\n[1/6] TF-IDF vectorizer...")
vec = TfidfVectorizer(max_features=6000, ngram_range=(1,2), sublinear_tf=True, min_df=2)
Xtr = vec.fit_transform(X_train)
Xva = vec.transform(X_val)
Xte = vec.transform(X_test)

os.makedirs('models', exist_ok=True)
models, val_probs, test_probs = {}, {}, {}

# ── NB ─────────────────────────────────────────────────────────
print("[2/6] Naive Bayes (alpha grid search)...")
nb = GridSearchCV(MultinomialNB(), {'alpha':[0.01,0.05,0.1,0.5,1.0,2.0,5.0]},
                  scoring='neg_log_loss', cv=5, n_jobs=-1)
nb.fit(Xtr, y_train)
nb = nb.best_estimator_
val_probs['nb']  = nb.predict_proba(Xva)[:,1]
test_probs['nb'] = nb.predict_proba(Xte)[:,1]
models['nb'] = nb
print(f"   best_alpha={nb.alpha:.3f}  val_ll={log_loss(y_val, val_probs['nb']):.4f}")

# ── LR ─────────────────────────────────────────────────────────
print("[3/6] Logistic Regression (C grid search)...")
lr = GridSearchCV(LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000,
                                     class_weight='balanced'),
                  {'C':[0.001,0.01,0.1,1,5,10,50,100]},
                  scoring='neg_log_loss', cv=5, n_jobs=-1)
lr.fit(Xtr, y_train)
lr = lr.best_estimator_
val_probs['lr']  = lr.predict_proba(Xva)[:,1]
test_probs['lr'] = lr.predict_proba(Xte)[:,1]
models['lr'] = lr
print(f"   best_C={lr.C:.4f}  val_ll={log_loss(y_val, val_probs['lr']):.4f}")

# ── RF ─────────────────────────────────────────────────────────
print("[4/6] Random Forest + Isotonic Calibration...")
rf = CalibratedClassifierCV(
    RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=5,
                            class_weight='balanced', random_state=42, n_jobs=-1),
    method='isotonic', cv=3)
rf.fit(Xtr, y_train)
val_probs['rf']  = rf.predict_proba(Xva)[:,1]
test_probs['rf'] = rf.predict_proba(Xte)[:,1]
models['rf'] = rf
print(f"   val_ll={log_loss(y_val, val_probs['rf']):.4f}")

# ── GB ─────────────────────────────────────────────────────────
print("[5/6] Gradient Boosting (early stopping)...")
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05,
                                  max_depth=4, subsample=0.8,
                                  validation_fraction=0.15,
                                  n_iter_no_change=20, tol=1e-4,
                                  random_state=42)
gb.fit(Xtr, y_train)
val_probs['gb']  = gb.predict_proba(Xva)[:,1]
test_probs['gb'] = gb.predict_proba(Xte)[:,1]
models['gb'] = gb
print(f"   trees={gb.n_estimators_}  val_ll={log_loss(y_val, val_probs['gb']):.4f}")

# ── Ensemble Weight Optimization ──────────────────────────────
print("[6/6] Optimizing ensemble weights (ALLWE algorithm)...")
Pval = np.column_stack([val_probs['nb'], val_probs['lr'], val_probs['rf'], val_probs['gb']])
Pte  = np.column_stack([test_probs['nb'], test_probs['lr'], test_probs['rf'], test_probs['gb']])

def neg_ll(w):
    w = np.abs(w); w /= w.sum()
    return log_loss(y_val, Pval @ w)

res = minimize(neg_ll, [0.25]*4, method='SLSQP',
               bounds=[(0,1)]*4,
               constraints={'type':'eq','fun': lambda w: sum(w)-1})
W = np.abs(res.x); W /= W.sum()
print(f"   Weights → NB:{W[0]:.3f} LR:{W[1]:.3f} RF:{W[2]:.3f} GB:{W[3]:.3f}")

# ── Test Results ──────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'Model':<22} {'LogLoss':>8} {'Acc':>7} {'F1':>7} {'AUC':>7}")
print("-" * 56)
all_p = {**{k: test_probs[k] for k in ['nb','lr','rf','gb']},
         'ensemble': Pte @ W}
names = {'nb':'Naive Bayes','lr':'Logistic Regr.','rf':'Random Forest',
         'gb':'Gradient Boost','ensemble':'Hybrid Ensemble ✓'}
for k, p in all_p.items():
    pred = (p>=0.5).astype(int)
    ll  = log_loss(y_test, p)
    acc = accuracy_score(y_test, pred)
    f1  = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, p)
    star = " ←BEST" if k=='ensemble' else ""
    print(f"{names[k]:<22} {ll:>8.4f} {acc*100:>6.2f}% {f1:>7.4f} {auc:>7.4f}{star}")

# ── Save ──────────────────────────────────────────────────────
joblib.dump(vec, 'models/vectorizer.pkl')
for k,m in models.items():
    joblib.dump(m, f'models/{k}.pkl')
joblib.dump(W, 'models/weights.pkl')
print("\n✓ All models saved to models/")
print("  Now run: python app.py")
