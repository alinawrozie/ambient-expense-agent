import pytest
from unittest.mock import AsyncMock, MagicMock
from google.genai import types

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

@pytest.fixture(autouse=True)
def mock_genai_client(monkeypatch):
    import google.genai as genai_module
    monkeypatch.setattr(genai_module, "Client", MockGenAIClient)
