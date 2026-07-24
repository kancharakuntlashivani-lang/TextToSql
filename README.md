# Shivani Text-to-SQL Research Application

A consistent Streamlit application for comparing relevant-schema strategies on two datasets:

- BIRD Mini-Dev
- Spider validation split

The application applies the same Full, Top-1, Top-3 and Top-5 experimental pattern to both datasets. It also trains three classifiers to predict exact-match success:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

## Research caution

The ML models predict experiment success from observed run features. They are secondary analytical models; they do not replace the Text-to-SQL generator. Do not report model results until enough real OpenAI experiment rows have been collected.

## Local setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
python -m streamlit run app.py
```

## Render deployment

1. Push this folder to GitHub.
2. In Render choose **New > Blueprint** and select the repository.
3. Add `OPENAI_API_KEY` as a secret environment variable.
4. Deploy.

The container binds Streamlit to `0.0.0.0` and Render's `$PORT`.

## Important API note

ChatGPT Plus does not include API billing. The app catches missing quota and authentication errors and displays a clear message instead of a traceback.
