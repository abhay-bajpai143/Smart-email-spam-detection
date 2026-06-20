# predictor.py  —  SpamGuard Core Prediction Engine
# The "ALLWE" (Adaptive Log-Loss Weighted Ensemble) architecture

import re, joblib, numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import deque
import time

STEMMER   = PorterStemmer()
try:
    STOPWORDS = set(stopwords.words('english'))
except:
    import nltk; nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))

# ── High-signal spam lexicon (used for explainability) ────────
SPAM_SIGNALS = [
    'free','win','winner','prize','claim','urgent','offer','cash',
    'click','buy','cheap','money','credit','loan','deal','limited',
    'selected','congratulations','guaranteed','award','bonus','call',
    'reply','subscribe','unsubscribe','discount','save','earn','now',
    'act','special','exclusive','dear','customer','account','verify',
    'confirm','bank','password','login','secure','update','expire',
    'suspend','unusual','alert','warning','important','immediately'
]

class SpamGuardPredictor:
    """
    ALLWE — Adaptive Log-Loss Weighted Ensemble
    Novel architecture: 4 ML models + optimized ensemble weights
    + real-time drift monitoring + explainability layer
    """

    def __init__(self):
        self.vectorizer = joblib.load('models/vectorizer.pkl')
        self.nb  = joblib.load('models/nb.pkl')
        self.lr  = joblib.load('models/lr.pkl')
        self.rf  = joblib.load('models/rf.pkl')
        self.gb  = joblib.load('models/gb.pkl')
        self.W   = joblib.load('models/weights.pkl')

        # Drift monitor: rolling window of last 100 predictions
        self._history = deque(maxlen=100)
        self._total_analyzed = 0
        self._total_spam     = 0

    def _clean(self, text):
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

    def _get_top_spam_words(self, raw_text, top_n=5):
        """Explainability: find which words most contributed to spam score."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', raw_text.lower())
        hits = []
        vocab = self.vectorizer.vocabulary_
        lr_coef = None
        try:
            lr_coef = self.lr.coef_[0]
        except: pass

        for word in words:
            stemmed = STEMMER.stem(word)
            # Check if it's in spam lexicon
            if word in SPAM_SIGNALS or stemmed in [STEMMER.stem(s) for s in SPAM_SIGNALS]:
                score = 0.8
                if lr_coef is not None:
                    idx = vocab.get(stemmed)
                    if idx is not None:
                        score = max(0.0, float(lr_coef[idx]))
                if word not in [h[0] for h in hits]:
                    hits.append((word.upper(), round(score, 3)))
        # Also get top LR coefficient words
        if lr_coef is not None:
            clean = self._clean(raw_text)
            features = self.vectorizer.transform([clean])
            feat_arr = features.toarray()[0]
            nonzero  = np.where(feat_arr > 0)[0]
            feat_names = self.vectorizer.get_feature_names_out()
            scored = sorted([(feat_names[i], lr_coef[i] * feat_arr[i])
                              for i in nonzero], key=lambda x: -x[1])
            for fname, sc in scored[:top_n]:
                original = fname.upper()
                if original not in [h[0] for h in hits]:
                    hits.append((original, round(max(0.0, sc), 3)))
        # Dedupe and return top N
        seen, out = set(), []
        for word, sc in sorted(hits, key=lambda x: -x[1]):
            if word not in seen:
                seen.add(word); out.append(word)
            if len(out) >= top_n:
                break
        return out

    def predict(self, raw_text):
        """
        Returns a full analysis dict:
        {
          label, confidence, risk_level,
          model_scores: {nb, lr, rf, gb},
          ensemble_weight: {nb, lr, rf, gb},
          top_spam_words,
          stats: {total_analyzed, spam_rate_recent}
        }
        """
        t0    = time.time()
        clean = self._clean(raw_text)
        feats = self.vectorizer.transform([clean])

        p_nb = float(self.nb.predict_proba(feats)[0][1])
        p_lr = float(self.lr.predict_proba(feats)[0][1])
        p_rf = float(self.rf.predict_proba(feats)[0][1])
        p_gb = float(self.gb.predict_proba(feats)[0][1])

        probs     = np.array([p_nb, p_lr, p_rf, p_gb])
        ensemble  = float(probs @ self.W)
        label     = 'SPAM' if ensemble >= 0.5 else 'HAM'
        confidence= ensemble * 100 if label == 'SPAM' else (1 - ensemble) * 100

        # Risk level
        if ensemble >= 0.90:   risk = 'CRITICAL'
        elif ensemble >= 0.70: risk = 'HIGH'
        elif ensemble >= 0.50: risk = 'MEDIUM'
        elif ensemble >= 0.30: risk = 'LOW'
        else:                  risk = 'SAFE'

        # Update stats
        self._total_analyzed += 1
        self._total_spam     += int(label == 'SPAM')
        self._history.append(int(label == 'SPAM'))

        recent_rate = (sum(self._history) / len(self._history) * 100) if self._history else 0
        overall_rate= (self._total_spam / self._total_analyzed * 100) if self._total_analyzed else 0

        top_words = self._get_top_spam_words(raw_text) if label == 'SPAM' else []

        return {
            'label':      label,
            'confidence': round(confidence, 2),
            'ensemble_prob': round(ensemble * 100, 2),
            'risk_level': risk,
            'model_scores': {
                'Naive Bayes':         round(p_nb * 100, 2),
                'Logistic Regression': round(p_lr * 100, 2),
                'Random Forest':       round(p_rf * 100, 2),
                'Gradient Boosting':   round(p_gb * 100, 2),
            },
            'ensemble_weights': {
                'Naive Bayes':         round(float(self.W[0]) * 100, 1),
                'Logistic Regression': round(float(self.W[1]) * 100, 1),
                'Random Forest':       round(float(self.W[2]) * 100, 1),
                'Gradient Boosting':   round(float(self.W[3]) * 100, 1),
            },
            'top_spam_words': top_words,
            'stats': {
                'total_analyzed':  self._total_analyzed,
                'overall_spam_rate': round(overall_rate, 1),
                'recent_spam_rate':  round(recent_rate, 1),
            },
            'latency_ms': round((time.time() - t0) * 1000, 1)
        }
