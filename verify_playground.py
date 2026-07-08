import json
import requests

BASE_URL = "http://127.0.0.1:8080"
USER_ID = "test_user_playground"
EXPENSE_PAYLOAD = {
    "amount": 150.0,
    "submitter": "alice@company.com",
    "category": "software",
    "description": "IDE License",
    "date": "2026-06-06"
}

# Create a session
session_id = requests.post(f"{BASE_URL}/apps/expense_agent/users/{USER_ID}/sessions").json()["id"]

# Run the agent stream
data = {
    "app_name": "expense_agent",
    "user_id": USER_ID,
    "session_id": session_id,
    "new_message": {
        "role": "user",
        "parts": [{"text": json.dumps(EXPENSE_PAYLOAD)}]
    },
    "streaming": True
}

res = requests.post(f"{BASE_URL}/run_sse", json=data, stream=True)

print("--- Printing raw SSE lines ---")
for line in res.iter_lines():
    if line:
        line_str = line.decode("utf-8")
        if line_str.startswith("data: "):
            event = json.loads(line_str[6:])
            # Filter non-ASCII characters from print
            clean_repr = repr(event).encode('ascii', errors='replace').decode('ascii')
            print(clean_repr[:200] + " ... " + clean_repr[-150:])
