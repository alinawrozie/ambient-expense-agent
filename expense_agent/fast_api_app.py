# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os

if os.getenv("INTEGRATION_TEST") == "TRUE":
    import google.auth
    from unittest.mock import MagicMock
    google.auth.default = MagicMock(return_value=(MagicMock(), "mock-project"))

from collections.abc import AsyncIterator

import google.auth
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.cloud import logging as google_cloud_logging

from expense_agent.app_utils import services
from expense_agent.app_utils.a2a import attach_a2a_routes
from expense_agent.app_utils.reasoning_engine_adapter import (
    attach_reasoning_engine_routes,
)
from expense_agent.app_utils.telemetry import (
    setup_agent_engine_telemetry,
    setup_telemetry,
)
from expense_agent.app_utils.typing import Feedback

load_dotenv()
setup_telemetry()
# Must run before get_fast_api_app to set the tracer provider resource.
setup_agent_engine_telemetry()
if os.getenv("INTEGRATION_TEST") == "TRUE" or os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() != "true":
    import logging as standard_logging
    os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("GOOGLE_CLOUD_PROJECT", "mock-project")
    os.environ["GOOGLE_CLOUD_LOCATION"] = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    import vertexai
    vertexai.init(project=os.environ["GOOGLE_CLOUD_PROJECT"], location=os.environ["GOOGLE_CLOUD_LOCATION"])
    project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
    logger = standard_logging.getLogger(__name__)
else:
    _, project_id = google.auth.default()
    logging_client = google_cloud_logging.Client()
    logger = logging_client.logger(__name__)


allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Runner for the A2A path, sharing the same session/artifact services as the
    # adk_api and reasoning_engine paths (see services.py). Imported here so the
    # agent is built after env/telemetry setup.
    from expense_agent.agent import app as adk_app
    from expense_agent.agent import root_agent

    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    # Shared by the A2A path and the reasoning_engine adapter routes.
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
    )
    yield


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=False,
    lifespan=lifespan,
)
app.title = "ambient-expense-agent"
app.description = "API for interacting with the Agent ambient-expense-agent"


# Proxy routes so the Vertex AI Console Playground (reasoning_engine SDK) can
# talk to this agent alongside the native adk_api routes.
attach_reasoning_engine_routes(app)


from fastapi import Request
from google.genai import types
import json


@app.post("/pubsub")
async def pubsub_trigger(request: Request) -> dict[str, str]:
    """Pub/Sub Push subscription endpoint that feeds events into the workflow."""
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse JSON body: {e}")
        return {"status": "error", "message": "Invalid JSON"}

    subscription_path = body.get("subscription", "default-subscription")
    # Normalize subscription path down to the final segment (short name)
    session_id = subscription_path.split("/")[-1]
    
    message_data = body.get("message", {})
    if not message_data:
        logger.warning("Pub/Sub message contains no message data.")
        return {"status": "ignored", "message": "No message data"}

    logger.info(f"Received Pub/Sub message for subscription: {session_id}")

    # Build the user message content from message data
    new_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=json.dumps(message_data))]
    )

    runner = app.state.runner
    events = []
    try:
        async for event in runner.run_async(
            user_id="pubsub_client",
            session_id=session_id,
            new_message=new_message,
        ):
            events.append(event)
    except Exception as e:
        logger.error(f"Error running workflow for session {session_id}: {e}")
        return {"status": "error", "message": str(e)}

    # Check if the workflow paused for human-in-the-loop review
    paused = False
    for event in events:
        if (
            event.content
            and event.content.parts
            and len(event.content.parts) > 0
            and event.content.parts[0].function_call
            and event.content.parts[0].function_call.name == "adk_request_input"
        ):
            paused = True
            break

    status_str = "paused_for_approval" if paused else "completed"
    logger.info(f"Workflow execution {status_str} for session {session_id}")
    return {"status": "success", "session_id": session_id, "workflow_state": status_str}


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    if hasattr(logger, "log_struct"):
        logger.log_struct(feedback.model_dump(), severity="INFO")
    else:
        logger.info(f"Feedback collected: {feedback.model_dump()}")
    return {"status": "success"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
