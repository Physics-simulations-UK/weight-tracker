# weight-tracker
Weight & Nutrition Tracker – Version 5

This version converts the tracker to Streamlit and adds Gemini-powered intelligent food analysis.

Files

app.py — complete Streamlit application

requirements.txt — Python dependencies

Streamlit secret

In Streamlit Community Cloud, add this secret in your app settings:

GEMINI_API_KEY = "your-key-here"

The app reads it with:

api_key = st.secrets.get("GEMINI_API_KEY", "")

For local development, use .streamlit/secrets.toml and do not commit that file to GitHub.

Run locally

pip install -r requirements.txt
streamlit run app.py

Current storage

Version 5 writes tracking data to tracker_data.json. This is intentionally simple for testing.
Use the built-in JSON export regularly. For a production hosted version, move persistence to a durable database such as Supabase/Postgres.

AI food workflow

Open AI food.

Enter a natural description such as:
180g chicken breast, 200g hash browns at 159 kcal per 100g, broccoli and carrots

Click Analyse food.

Review and edit the structured result.

Add it to Breakfast, Lunch, Dinner or Snacks.

Optionally save regular foods to the Saved Foods catalogue.

Gemini is configured to prefer:

exact nutrition information supplied in the message;

saved-food values;

web-researched values using Google Search;

conservative estimates as a fallback.
