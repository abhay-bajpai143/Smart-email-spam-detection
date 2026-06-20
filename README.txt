================================================================
  SpamGuard — Setup & Run Guide
================================================================

STEP 1 — Install (run once)
───────────────────────────
  Windows: Double-click STEP1_install.bat
  Mac/Linux: pip install flask scikit-learn pandas numpy nltk joblib scipy matplotlib seaborn
             python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

STEP 2 — Train Models (run once, takes 5–10 min)
──────────────────────────────────────────────────
  python STEP2_train.py

  Downloads dataset automatically. Trains NB, LR, RF, GB.
  Optimizes ensemble weights. Saves to models/ folder.

STEP 3 — Launch Web App
────────────────────────
  python app.py

  Open browser: http://localhost:5000

STEP 4 — Use It
────────────────
  Paste any email text → Click ANALYZE EMAIL
  Or click Quick Test Samples to load examples
  Ctrl+Enter also works to analyze

================================================================
  FILES
================================================================

  STEP1_install.bat     Install dependencies (Windows)
  STEP2_train.py        Train all 4 models + optimize weights
  predictor.py          Core ALLWE prediction engine
  app.py                Flask web application
  templates/index.html  Web UI
  PATENT_DRAFT.txt      Patent application draft (7 claims)

================================================================
