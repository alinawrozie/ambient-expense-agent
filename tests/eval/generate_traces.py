import json
import os
import asyncio
from unittest.mock import MagicMock
import google.genai
from google.genai import types

# Patch google.genai.Client globally so trace generation is offline-capable and deterministic
def get_mock_response():
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[
                        types.Part(
                            text='{"risk_level": "LOW", "risk_factors": [], "summary": "Scrubbed expense description looks standard and low risk."}'
                        )
                    ],
                    role="model"
                )
            )
        ],
        model_version="gemini-3.1-flash-lite",
    )

class MockModels:
    def generate_content(self, *args, **kwargs):
        return get_mock_response()
    async def generate_content_async(self, *args, **kwargs):
        return get_mock_response()

class MockAioModels:
    async def generate_content(self, *args, **kwargs):
        return get_mock_response()
    async def generate_content_stream(self, *args, **kwargs):
        async def generator():
            yield get_mock_response()
        return generator()

class MockAio:
    def __init__(self):
        self.models = MockAioModels()

class MockGenAIClient:
    def __init__(self, *args, **kwargs):
        self.vertexai = False
        self.models = MockModels()
        self.aio = MockAio()

google.genai.Client = MockGenAIClient

class DummyFunctionResponse:
    def __init__(self, id, response):
        self.id = id
        self.response = response

class DummyPart:
    def __init__(self, function_response):
        self.function_response = function_response
        self.text = None

class DummyContent:
    def __init__(self, parts):
        self.parts = parts
        self.role = "user"

# Now import ADK runner and agent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from expense_agent.agent import root_agent

async def main():
    dataset_path = "tests/eval/datasets/basic-dataset.json"
    output_dir = "artifacts/traces"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated_traces.json")
    
    with open(dataset_path, "r") as f:
        data = json.load(f)
        
    eval_cases = data.get("eval_cases", [])
    traces = []
    
    for case in eval_cases:
        case_id = case.get("eval_case_id")
        prompt_content = case.get("prompt")
        reference = case.get("reference")
        
        prompt_text = prompt_content["parts"][0]["text"]
        
        session_service = InMemorySessionService()
        runner = Runner(agent=root_agent, session_service=session_service, app_name="expense_agent")
        session = await session_service.create_session(user_id="eval_user", app_name="expense_agent")
        
        # Turn 1: Send the user prompt
        message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt_text)]
        )
        
        events_turn_1 = []
        paused = False
        async for event in runner.run_async(
            new_message=message,
            user_id="eval_user",
            session_id=session.id
        ):
            events_turn_1.append(event)
            if (
                event.content
                and event.content.parts
                and len(event.content.parts) > 0
                and event.content.parts[0].function_call
                and event.content.parts[0].function_call.name == "adk_request_input"
            ):
                paused = True
        
        events_turn_2 = []
        final_response_text = ""
        
        if paused:
            # Determine human decision:
            # If it contains prompt injection -> reject
            # Otherwise -> approve
            is_injection = "Ignore all instructions" in prompt_text or "Bypass all rules" in prompt_text
            decision = "reject" if is_injection else "approve"
            
            # Send the human decision turn as a FunctionResponse to resume the RequestInput
            fr = DummyFunctionResponse(id="approval_decision", response={"result": decision})
            part = DummyPart(function_response=fr)
            resume_message = DummyContent(parts=[part])
            
            async for event in runner.run_async(
                new_message=resume_message,
                user_id="eval_user",
                session_id=session.id
            ):
                events_turn_2.append(event)
                if event.output:
                    final_response_text = str(event.output)
        else:
            # Auto-approved (under $100)
            for event in events_turn_1:
                if event.output:
                    final_response_text = str(event.output)
                    
        # Construct the trace structure
        turns = []
        
        # Turn 1
        serialized_events_1 = []
        serialized_events_1.append({
            "author": "user",
            "content": {
                "role": "user",
                "parts": [{"text": prompt_text}]
            }
        })
        for event in events_turn_1:
            try:
                evt_dict = json.loads(event.model_dump_json(exclude_none=True))
                # Normalize author
                author = "agent" if not event.content or event.content.role == "model" else "user"
                serialized_events_1.append({
                    "author": author,
                    "content": evt_dict.get("content", {})
                })
            except Exception:
                pass
        turns.append({
            "turn_index": 0,
            "events": serialized_events_1
        })
        
        if paused:
            # Turn 2
            serialized_events_2 = []
            decision = "reject" if is_injection else "approve"
            serialized_events_2.append({
                "author": "user",
                "content": {
                    "role": "user",
                    "parts": [{"text": decision}]
                }
            })
            for event in events_turn_2:
                try:
                    evt_dict = json.loads(event.model_dump_json(exclude_none=True))
                    author = "agent" if not event.content or event.content.role == "model" else "user"
                    serialized_events_2.append({
                        "author": author,
                        "content": evt_dict.get("content", {})
                    })
                except Exception:
                    pass
            turns.append({
                "turn_index": 1,
                "events": serialized_events_2
            })
            
        trace = {
            "eval_case_id": case_id,
            "prompt": {
                "role": "user",
                "parts": [{"text": prompt_text}]
            },
            "responses": [
                {
                    "response": {
                        "role": "model",
                        "parts": [{"text": final_response_text or "PAUSED_PENDING_APPROVAL"}]
                    }
                }
            ],
            "reference": {
                "response": {
                    "role": "model",
                    "parts": [{"text": str(reference)}]
                }
            } if reference else None,
            "agent_data": {
                "turns": turns
            }
        }
        traces.append(trace)
        print(f"Generated trace for case: {case_id}")
        
    with open(output_path, "w") as f:
        json.dump({"eval_cases": traces}, f, indent=2)
    print(f"Traces written to: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
