# Medley ML Backend

This is the FastAPI-based ML backend for Medley, providing AI-powered medicine and shop recommendations.

## Local Development

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

## Deployment on Render

1. **Push your backend folder to a GitHub repo.**
2. **Create a new Web Service on [Render](https://render.com/):**
   - Environment: Python 3.11+
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port 10000`
   - (You can use any port, but 10000 is Render's default for Python)
3. **Set environment variables:**
   - If you use `.env`, add the same variables in Render's dashboard under Environment > Environment Variables.
4. **Expose port 10000** (or the port you set in Start Command).
5. **(Optional) Add a `render.yaml` for infrastructure-as-code deployments.**

## Required Files
- `requirements.txt`
- `main.py`
- `services/` (your Python modules)
- `data/` (your JSON data files)

## .env Example
```
# Example .env
# (Set these in Render dashboard for production)

# Any custom variables you use
```

## Notes
- Make sure your FastAPI app is named `app` in `main.py`.
- If you use large ML models, consider Render's higher memory plans.
- For persistent data, use a managed database (not local JSON files).
