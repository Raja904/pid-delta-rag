"""
chat/answer.py
Grounded answer generation using LangGraph ReAct Agent.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
import os

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from src.chat.tools import semantic_search, exact_tag_search
from src.observability.tracer import trace_stage
from src.observability.logger import log

@dataclass
class Citation:
    source: str
    ref:    str
    text:   str

@dataclass
class GroundedAnswer:
    question:  str
    answer:    str
    citations: list[Citation]
    model:     str

def answer(
    question: str,
    summary_dict: dict = None,
    collection_name: str = "delta_chat",
    n_results: int = 6,
    model: str | None = None,
) -> GroundedAnswer:
    """Uses a ReAct agent to answer the question using tools."""
    
    with trace_stage("agent_setup"):
        model_name = model or os.getenv("LLM_MODEL", "qwen/qwen3.6-27b")
        llm = ChatGroq(model=model_name, temperature=0.1)
        tools = [semantic_search, exact_tag_search]
        
        # Build the dynamic summary string
        summary_str = "No delta summary available."
        if summary_dict:
            summary_str = (
                f"Total Changes: {summary_dict.get('total', 0)}\n"
                f"Added: {summary_dict.get('added', 0)}\n"
                f"Removed: {summary_dict.get('removed', 0)}\n"
                f"Modified: {summary_dict.get('modified', 0)}"
            )
            
        system_prompt = f"""You are a highly capable technical document analyst agent.
You help engineers understand differences between two document revisions (PID A = base, PID B = revised) and a delta report.

OVERALL DELTA SUMMARY FOR THIS SESSION:
{summary_str}

RULES:
1. You have access to two tools: `semantic_search` (for general queries) and `exact_tag_search` (for specific Component IDs, Line Numbers, Valves, Dimensions).
2. You MUST use these tools to find information before answering. Do NOT hallucinate.
3. Every factual claim in your final answer MUST reference a citation exactly as it appears in the tool output (e.g., [PID_A·p.N·bID] or [DELTA·change-ID]).
4. Be concise and precise.
5. If the user provides a simple greeting (like "hi", "hello"), respond politely and ask how you can help with the documents. Do not use tools for greetings.
"""
        
        agent_executor = create_react_agent(llm, tools, prompt=system_prompt)
    
    with trace_stage("agent_invoke"):
        log.info("Invoking ReAct agent", extra={"query": question[:80]})
        result = agent_executor.invoke({"messages": [("user", question)]})
        final_answer = result["messages"][-1].content
        
        # Remove <think>...</think> reasoning blocks from Qwen/DeepSeek models just in case
        final_answer = re.sub(r'<think>.*?</think>\s*', '', final_answer, flags=re.DOTALL).strip()
        
        # Reconstruct citations from the Tool messages
        citations_list = []
        seen_refs = set()
        for msg in result["messages"]:
            if getattr(msg, "type", "") == "tool":
                # Find all [ref]: text patterns in the tool output
                matches = re.finditer(r'^\[(.*?)\]:\s*(.*?)(?=\n^\[|\Z)', msg.content, flags=re.MULTILINE | re.DOTALL)
                for m in matches:
                    ref = m.group(1).strip()
                    text = m.group(2).strip()
                    if ref not in seen_refs:
                        source = "unknown"
                        if ref.startswith("PID_A"): source = "pid_a"
                        elif ref.startswith("PID_B"): source = "pid_b"
                        elif ref.startswith("DELTA"): source = "delta_report"
                        citations_list.append(Citation(source=source, ref=ref, text=text))
                        seen_refs.add(ref)

    return GroundedAnswer(
        question  = question,
        answer    = final_answer,
        citations = citations_list,
        model     = model_name
    )
