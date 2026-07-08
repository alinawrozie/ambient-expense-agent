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

import base64
import json
import re
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import Workflow, START, node
from google.genai import types
from pydantic import BaseModel, Field

from expense_agent.config import MODEL_NAME, THRESHOLD


# =====================================================================
# 1. Pydantic Schemas
# =====================================================================

class RiskAssessment(BaseModel):
    """Structured response for risk evaluation of the expense."""
    risk_level: str = Field(description="The evaluated risk level: LOW, MEDIUM, or HIGH")
    risk_factors: list[str] = Field(description="List of specific identified risk factors or policy issues")
    summary: str = Field(description="A brief summary explaining the risk assessment details")


# =====================================================================
# 2. Workflow Nodes
# =====================================================================

def extract_expense(ctx: Context, node_input: Any) -> Event:
    """Parses Pub/Sub base64 or plain JSON expense payload and extracts details."""
    print(f"[extract_expense] Raw input type: {type(node_input)}")
    print(f"[extract_expense] Raw input content: {node_input}")

    # Step A: Normalize node_input to a dictionary or extract from types.Content
    raw_dict = {}
    if isinstance(node_input, dict):
        raw_dict = node_input
    elif isinstance(node_input, str):
        try:
            raw_dict = json.loads(node_input)
        except json.JSONDecodeError:
            raw_dict = {"data": node_input}
    elif hasattr(node_input, "parts") and node_input.parts:
        parts_text = "".join(
            part.text for part in node_input.parts if hasattr(part, "text") and part.text
        )
        try:
            raw_dict = json.loads(parts_text)
        except json.JSONDecodeError:
            raw_dict = {"data": parts_text}
    else:
        try:
            raw_dict = json.loads(str(node_input))
        except Exception:
            raw_dict = {"data": node_input}

    # Step B: Locate the "data" payload
    data_val = raw_dict.get("data")
    if data_val is None:
        if any(k in raw_dict for k in ["amount", "submitter", "category", "description"]):
            data_val = raw_dict
        else:
            raise ValueError(f"No valid expense data found in input payload: {raw_dict}")

    # Step C: Parse base64 string, plain JSON string, or dictionary
    parsed_data = {}
    if isinstance(data_val, str):
        try:
            # Try decoding as base64
            decoded_bytes = base64.b64decode(data_val.encode('utf-8'))
            decoded_str = decoded_bytes.decode('utf-8')
            parsed_data = json.loads(decoded_str)
            print("[extract_expense] Decoded and parsed base64 data.")
        except Exception:
            try:
                # Fallback: try parsing as raw JSON string
                parsed_data = json.loads(data_val)
                print("[extract_expense] Parsed JSON string data.")
            except json.JSONDecodeError:
                raise ValueError(f"String data is not valid JSON or base64 JSON: {data_val}")
    elif isinstance(data_val, dict):
        parsed_data = data_val
        print("[extract_expense] Extracted payload from dictionary.")
    else:
        raise ValueError(f"Unexpected data field type: {type(data_val)}")

    # Step D: Construct normalized expense dictionary
    try:
        amount = float(parsed_data.get("amount", 0.0))
    except (ValueError, TypeError):
        amount = 0.0

    expense = {
        "amount": amount,
        "submitter": str(parsed_data.get("submitter", "Unknown")),
        "category": str(parsed_data.get("category", "General")),
        "description": str(parsed_data.get("description", "")),
        "date": str(parsed_data.get("date", ""))
    }

    print(f"[extract_expense] Parsed expense details: {expense}")
    return Event(output=expense, state={"expense": expense})


def route_expense(ctx: Context, node_input: dict) -> Event:
    """Evaluates the expense amount against the threshold to route the flow."""
    amount = node_input["amount"]
    print(f"[route_expense] Amount: ${amount} | Threshold: ${THRESHOLD}")
    if amount < THRESHOLD:
        return Event(output=node_input, route="auto_approve")
    else:
        return Event(output=node_input, route="llm_review")


def auto_approve(ctx: Context, node_input: dict) -> dict:
    """Instantly auto-approves expenses below the configured threshold."""
    decision = {
        "status": "APPROVED",
        "reason": f"Auto-approved: amount ${node_input['amount']} is below the ${THRESHOLD} threshold.",
        "expense": node_input,
        "risk_assessment": None
    }
    print(f"[auto_approve] Instant decision: {decision['reason']}")
    return decision


# =====================================================================
# Security Controls: PII Scrubbing and Prompt Injection Defense
# =====================================================================

SSN_REGEX_HYPHEN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
SSN_REGEX_RAW = re.compile(r'\b\d{9}\b')
CC_REGEX_HYPHEN = re.compile(r'\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b')
CC_REGEX_RAW = re.compile(r'\b\d{16}\b')

INJECTION_KEYWORDS = [
    "ignore previous instructions",
    "ignore all instructions",
    "system prompt",
    "you are now",
    "override the rules",
    "bypass the rules",
    "auto-approve",
    "auto approve",
    "always approve",
    "instantly approve"
]


def scrub_pii(text: str) -> tuple[str, list[str]]:
    """Scrubs Social Security Numbers and Credit Card numbers from the text."""
    redacted = []
    
    # Scrub credit cards first
    text, count_cc_hyphen = CC_REGEX_HYPHEN.subn("[REDACTED CREDIT CARD]", text)
    text, count_cc_raw = CC_REGEX_RAW.subn("[REDACTED CREDIT CARD]", text)
    if count_cc_hyphen > 0 or count_cc_raw > 0:
        redacted.append("Credit Card")
        
    # Scrub SSNs
    text, count_ssn_hyphen = SSN_REGEX_HYPHEN.subn("[REDACTED SSN]", text)
    text, count_ssn_raw = SSN_REGEX_RAW.subn("[REDACTED SSN]", text)
    if count_ssn_hyphen > 0 or count_ssn_raw > 0:
        redacted.append("SSN")
        
    return text, redacted


def detect_injection(text: str) -> bool:
    """Detects simple prompt injection patterns in the text."""
    normalized = text.lower()
    for kw in INJECTION_KEYWORDS:
        if kw in normalized:
            return True
    return False


def security_checkpoint(ctx: Context, node_input: dict) -> Event:
    """Security filter to scrub PII and defend against prompt injections."""
    expense = node_input.copy()
    description = expense.get("description", "")
    
    # 1. Scrub personal data
    scrubbed_desc, redacted_categories = scrub_pii(description)
    expense["description"] = scrubbed_desc
    
    # Keep the state updated with the scrubbed expense details
    ctx.state["expense"] = expense
    if redacted_categories:
        ctx.state["redactions"] = redacted_categories
        print(f"[security_checkpoint] Redacted categories: {redacted_categories}")
    
    # 2. Defend against prompt injection
    if detect_injection(scrubbed_desc):
        print("[security_checkpoint] Prompt injection detected! Routing to manual review.")
        risk_assessment = RiskAssessment(
            risk_level="HIGH",
            risk_factors=["POTENTIAL PROMPT INJECTION DETECTED"],
            summary=(
                "The system detected phrases in the description matching known prompt injection "
                "patterns. LLM analysis was bypassed for security, and the expense has been flagged "
                "for immediate manual auditing."
            )
        )
        return Event(
            output=risk_assessment,
            route="security_bypass",
            state={"expense": expense, "redactions": redacted_categories}
        )
        
    # Clean flow: proceed to LLM review
    return Event(
        output=expense,
        route="clean",
        state={"expense": expense, "redactions": redacted_categories}
    )


# LLM node: Evaluates risk factors using the configured Gemini model
llm_review_agent = LlmAgent(
    name="llm_review",
    model=MODEL_NAME,
    instruction=(
        "You are an expense policy and risk auditor. Review the details of the submitted "
        "expense report (including submitter, category, amount, description, and date) "
        "for policy violations, fraud risk, duplicates, or anomalies. Provide a structured risk "
        "assessment matching the RiskAssessment schema."
    ),
    output_schema=RiskAssessment,
)


@node(rerun_on_resume=True)
async def human_approval(ctx: Context, node_input: RiskAssessment) -> Event:
    """Pauses workflow to request human review, and handles the decision upon resumption."""
    # Check if this node has been resumed with user input
    if not ctx.resume_inputs or "approval_decision" not in ctx.resume_inputs:
        expense = ctx.state["expense"]
        prompt_message = (
            f"\n🔔 HUMAN APPROVAL REQUIRED 🔔\n"
            f"An expense of ${expense['amount']} submitted by {expense['submitter']} is "
            f"above the ${THRESHOLD} limit and requires manual approval.\n"
            f"Category: {expense['category']} | Date: {expense['date']}\n"
            f"Description: {expense['description']}\n\n"
            f"🔍 RISK ASSESSMENT FINDINGS:\n"
            f"• Risk Level: {node_input.risk_level}\n"
            f"• Risk Factors: {', '.join(node_input.risk_factors) if node_input.risk_factors else 'None'}\n"
            f"• Risk Summary: {node_input.summary}\n\n"
            f"Do you approve or reject this expense? (Type 'approve' or 'reject')"
        )
        print("[human_approval] Yielding RequestInput to pause the workflow.")
        yield RequestInput(
            interrupt_id="approval_decision",
            message=prompt_message
        )
        return

    # Extract the user's decision
    decision = ctx.resume_inputs["approval_decision"].strip().lower()
    is_approved = "approve" in decision
    
    status = "APPROVED" if is_approved else "REJECTED"
    reason = f"Decision by human reviewer: '{ctx.resume_inputs['approval_decision']}'."
    
    result = {
        "status": status,
        "reason": reason,
        "expense": ctx.state["expense"],
        "risk_assessment": node_input.model_dump()
    }
    print(f"[human_approval] Resumed! Human decision: {status}")
    yield Event(output=result)


def record_outcome(ctx: Context, node_input: dict) -> Event:
    """Logs the final outcome of the expense workflow."""
    print(f"[record_outcome] Final Result: {node_input['status']} - {node_input['reason']}")
    return Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=json.dumps(node_input))]
        ),
        output=node_input
    )


# =====================================================================
# 3. Workflow Graph Setup
# =====================================================================

root_agent = Workflow(
    name="expense_approval_workflow",
    description="Ambient expense approval workflow with auto-approval and LLM/HITL review.",
    edges=[
        (START, extract_expense),
        (extract_expense, route_expense),
        # Conditional routing from route_expense using a RoutingMap (dict)
        (route_expense, {
            "auto_approve": auto_approve,
            "llm_review": security_checkpoint
        }),
        # Branch A: Below threshold (Auto-approval) finishes
        (auto_approve, record_outcome),
        # Branch B: Above threshold (Security Checkpoint + LLM review / Direct HITL)
        (security_checkpoint, {
            "clean": llm_review_agent,
            "security_bypass": human_approval
        }),
        (llm_review_agent, human_approval),
        (human_approval, record_outcome),
    ]
)

app = App(
    root_agent=root_agent,
    name="expense_agent",
)
