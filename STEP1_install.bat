@echo off
echo ================================================
echo   SpamGuard - Installing Dependencies
echo ================================================
echo.
pip install flask scikit-learn pandas numpy nltk joblib scipy matplotlib seaborn
echo.
echo Downloading NLTK language data...
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
echo.
echo ================================================
echo   Done! Now run: python STEP2_train.py
echo   Then run:      python app.py
echo   Then open:     http://localhost:5000
echo ================================================
pause
