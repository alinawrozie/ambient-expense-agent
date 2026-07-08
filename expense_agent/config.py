import os

# Configurable dollar threshold for auto-approval vs human-in-the-loop approval
THRESHOLD = float(os.getenv("EXPENSE_THRESHOLD", "100.0"))

# Gemini model used for expense risk evaluation
MODEL_NAME = os.getenv("EXPENSE_MODEL", "gemini-3.1-flash-lite")
