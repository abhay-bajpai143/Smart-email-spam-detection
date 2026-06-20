# app.py  —  SpamGuard Web Application
# Run: python app.py
# Then open: http://localhost:5000

from flask import Flask, render_template, request, jsonify
from predictor import SpamGuardPredictor
import json, os

app = Flask(__name__)

# Load predictor once at startup
print("Loading SpamGuard models...")
predictor = SpamGuardPredictor()
print("Ready! Visit http://localhost:5000")

# ── Session history (in-memory) ───────────────────────────────
history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    result = predictor.predict(text)

    # Store in history (keep last 50)
    history.append({
        'text':    text[:80] + ('...' if len(text) > 80 else ''),
        'label':   result['label'],
        'conf':    result['confidence'],
        'risk':    result['risk_level']
    })
    if len(history) > 50:
        history.pop(0)

    result['history'] = history[-10:][::-1]   # last 10, newest first
    return jsonify(result)

@app.route('/bulk')
def bulk():
    return render_template('bulk.html')

@app.route('/bulk_analyze', methods=['POST'])
def bulk_analyze():
    data    = request.get_json()
    emails  = data.get('emails', [])
    if not emails:
        return jsonify({'error': 'No emails provided'}), 400

    results = []
    for i, text in enumerate(emails):
        text = text.strip()
        if not text:
            continue
        r = predictor.predict(text)
        results.append({
            'id':         i + 1,
            'preview':    text[:90] + ('...' if len(text) > 90 else ''),
            'full_text':  text,
            'label':      r['label'],
            'risk':       r['risk_level'],
            'confidence': r['confidence'],
            'ensemble_prob': r['ensemble_prob'],
            'top_spam_words': r['top_spam_words'],
            'model_scores':  r['model_scores'],
        })

    # Sort by ensemble spam probability descending
    results.sort(key=lambda x: x['ensemble_prob'], reverse=True)

    # Group by risk level
    groups = {'CRITICAL': [], 'HIGH': [], 'MEDIUM': [], 'LOW': [], 'SAFE': []}
    for r in results:
        groups[r['risk']].append(r)

    summary = {
        'total':    len(results),
        'spam':     sum(1 for r in results if r['label'] == 'SPAM'),
        'ham':      sum(1 for r in results if r['label'] == 'HAM'),
        'critical': len(groups['CRITICAL']),
        'high':     len(groups['HIGH']),
        'medium':   len(groups['MEDIUM']),
        'low':      len(groups['LOW']),
        'safe':     len(groups['SAFE']),
    }

    return jsonify({'summary': summary, 'groups': groups, 'all': results})

@app.route('/stats')
def stats():
    return jsonify(predictor.predict('')['stats'])

if __name__ == '__main__':
    if not os.path.exists('models/vectorizer.pkl'):
        print("ERROR: Models not found. Run STEP2_train.py first!")
    else:
        app.run(debug=False, port=5000)
