# wsgi.py
# This file serves as an entry point for Gunicorn

# Import your Flask app - adjust the import path as needed
from app import app

# This allows Gunicorn to find your app
if __name__ == "__main__":
    app.run()
