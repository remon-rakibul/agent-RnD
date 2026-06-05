"""
Multi-Agent System with LangGraph — Supervisor Pattern

This demonstrates a multi-agent architecture where a supervisor agent
routes tasks to specialized worker agents, compared to the single-agent
approach in agent.py.

Architecture:
  ┌────────────┐
  │ Supervisor │  ← Decides which worker to call (or finish)
  └─────┬──────┘
        │
   ┌────┴────┐
   ▼         ▼
┌──────┐  ┌──────────┐
│ Math │  │ Research  │  ← Specialized worker agents
│ Agent│  │  Agent    │
└──────┘  └──────────┘

Key differences from agent.py (single agent):
  - Single agent: One LLM with ALL tools bound, decides everything itself
  - Multi-agent: Supervisor routes to specialized agents, each with their
    own tools and system prompts — better separation of concerns
"""

from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langchain.messages import SystemMessage, HumanMessage, ToolMessage, AnyMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool
from typing import Literal
import operator
import json

# ==============================================================================
# Step 1: Define the shared LLM
# ==============================================================================

model = ChatOllama(
    model="gemma4:latest",
    temperature=0,
    num_ctx=2048,  # Reduce context size for faster responses
)

# ==============================================================================
# Step 2: Define tools for each specialized agent
# ==============================================================================

# --- Math Agent Tools ---
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# --- Research Agent Tools ---
@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a topic.

    Args:
        query: The search query
    """
    # Simulated search — replace with real API call in production
    return f"Wikipedia result for '{query}': This is a simulated search result. In production, this would call the Wikipedia API."


@tool
def get_current_date() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Group tools by agent
math_tools = [add, multiply, divide]
research_tools = [search_wikipedia, get_current_date]

math_tools_by_name = {t.name: t for t in math_tools}
research_tools_by_name = {t.name: t for t in research_tools}

# Create tool-augmented models for each agent
math_model = model.bind_tools(math_tools)
research_model = model.bind_tools(research_tools)

# ==============================================================================
# Step 3: Define the shared state
# ==============================================================================

class MultiAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    next_agent: str          # Which agent the supervisor routes to next
    current_agent: str       # Track which agent is currently active
    supervisor_visits: int  # Track cycle count to prevent infinite loops
    math_done: bool         # Track if math task completed
    research_done: bool     # Track if research task completed


# ==============================================================================
# Step 4: Define the Supervisor (Router) - Simple Keyword-Based
# ==============================================================================

def supervisor_node(state: MultiAgentState):
    """Supervisor decides which specialized agent to call next"""

    # Prevent infinite loops
    visits = state.get("supervisor_visits", 0)
    if visits >= 10:
        print("\n🛑 Supervisor: Max visits reached, finishing...")
        return {
            "next_agent": "FINISH",
            "current_agent": "supervisor",
            "supervisor_visits": visits + 1,
            "math_done": state.get("math_done", False),
            "research_done": state.get("research_done", False),
        }

    # Simple keyword-based routing (more reliable with Ollama)
    last_message = state["messages"][-1].content.lower() if state["messages"] else ""

    # Check if tasks are already done
    math_done = state.get("math_done", False)
    research_done = state.get("research_done", False)

    # Detect math keywords (if not done yet)
    math_keywords = ["multiply", "multiplied", "times", "plus", "minus", "add", "subtract", "divide",
                     "calculation", "calculate", "math", "arithmetic", "3 +", "4 +", "15 *", "7 *"]
    has_math = any(kw in last_message for kw in math_keywords)

    # Detect research/date keywords (if not done yet)
    research_keywords = ["date", "today", "time", "wikipedia", "search", "what is"]
    has_research = any(kw in last_message for kw in research_keywords)

    next_agent = "FINISH"
    reasoning = "No matching task found"

    if not math_done and has_math:
        next_agent = "math_agent"
        reasoning = "Detected math-related query"
    elif not research_done and has_research:
        next_agent = "research_agent"
        reasoning = "Detected research/date-related query"
    elif math_done and research_done:
        next_agent = "FINISH"
        reasoning = "All tasks completed"

    print(f"\n🔀 Supervisor Decision: route to '{next_agent}' — {reasoning}")

    return {
        "next_agent": next_agent,
        "current_agent": "supervisor",
        "supervisor_visits": visits + 1,
        "math_done": math_done,
        "research_done": research_done,
    }


# ==============================================================================
# Step 5: Define Worker Agent Nodes (each is its own tool-calling sub-loop)
# ==============================================================================

def math_agent_node(state: MultiAgentState):
    """Math agent: handles arithmetic using add, multiply, divide tools"""

    print("\n🧮 Math Agent is working...")

    # The math agent runs its own internal tool-calling loop
    messages = list(state["messages"])
    system_msg = SystemMessage(
        content="You are a specialized math agent. You can add, multiply, and divide numbers. "
                "Use the available tools to perform calculations. Be precise and show your work."
    )

    # LLM call
    response = math_model.invoke([system_msg] + messages)
    messages_to_add = [response]

    # If the LLM wants to call tools, execute them in a loop (max 3 iterations to prevent infinite loops)
    tool_iterations = 0
    while response.tool_calls and tool_iterations < 3:
        tool_iterations += 1
        tool_results = []
        for tool_call in response.tool_calls:
            t = math_tools_by_name[tool_call["name"]]
            observation = t.invoke(tool_call["args"])
            tool_results.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )
        messages_to_add.extend(tool_results)

        # Call the LLM again with the tool results
        response = math_model.invoke([system_msg] + messages + messages_to_add)
        messages_to_add.append(response)

    return {
        "messages": messages_to_add,
        "current_agent": "math_agent",
        "math_done": True,  # Mark math task as complete
        "research_done": state.get("research_done", False),
    }


def research_agent_node(state: MultiAgentState):
    """Research agent: handles general knowledge and search queries"""

    print("\n🔍 Research Agent is working...")

    messages = list(state["messages"])
    system_msg = SystemMessage(
        content="You are a specialized research agent. You can search Wikipedia for information "
                "and get the current date/time. Provide clear, informative answers."
    )

    # LLM call
    response = research_model.invoke([system_msg] + messages)
    messages_to_add = [response]

    # If the LLM wants to call tools, execute them in a loop (max 3 iterations to prevent infinite loops)
    tool_iterations = 0
    while response.tool_calls and tool_iterations < 3:
        tool_iterations += 1
        tool_results = []
        for tool_call in response.tool_calls:
            t = research_tools_by_name[tool_call["name"]]
            observation = t.invoke(tool_call["args"])
            tool_results.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )
        messages_to_add.extend(tool_results)

        # Call the LLM again with the tool results
        response = research_model.invoke([system_msg] + messages + messages_to_add)
        messages_to_add.append(response)

    return {
        "messages": messages_to_add,
        "current_agent": "research_agent",
        "math_done": state.get("math_done", False),
        "research_done": True,  # Mark research task as complete
    }


# ==============================================================================
# Step 6: Define routing logic
# ==============================================================================

def route_after_supervisor(state: MultiAgentState) -> Literal["math_agent", "research_agent", END]:
    """Route to the appropriate worker agent based on the supervisor's decision"""
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return END
    return next_agent


def route_after_worker(state: MultiAgentState) -> Literal["supervisor", END]:
    """After worker completes, route back to supervisor"""
    return "supervisor"


# ==============================================================================
# Step 7: Build the multi-agent graph
# ==============================================================================

builder = StateGraph(MultiAgentState)

# Add nodes
builder.add_node("supervisor", supervisor_node)
builder.add_node("math_agent", math_agent_node)
builder.add_node("research_agent", research_agent_node)

# Add edges
builder.add_edge(START, "supervisor")

# Supervisor routes to a worker or ends
builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    ["math_agent", "research_agent", END],
)

# After each worker finishes, go back to supervisor to decide next step
builder.add_edge("math_agent", "supervisor")
builder.add_edge("research_agent", "supervisor")

# Compile the multi-agent system
multi_agent = builder.compile()


# ==============================================================================
# Step 8: Visualize the graph
# ==============================================================================

multi_agent.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API,
    output_file_path="multi_agent_graph.png",
)
print("📊 Graph saved to multi_agent_graph.png")


# ==============================================================================
# Step 9: Run the multi-agent system
# ==============================================================================

print("\n" + "=" * 60)
print("🚀 Multi-Agent System — Supervisor Pattern")
print("=" * 60)

# Test with a multi-part request that requires both agents
messages = [HumanMessage(
    content="What is 15 multiplied by 7? Also, what is today's date?"
)]

initial_state = {
    "messages": messages,
    "supervisor_visits": 0,
    "math_done": False,
    "research_done": False,
}

print("\n🚀 Running multi-agent system...")
result = None
for result in multi_agent.stream(initial_state, stream_mode="values"):
    node_name = result.get("current_agent", "unknown")
    print(f"\n📍 Completed node: {node_name}")

print("\n" + "=" * 60)
print("📝 Full Conversation:")
print("=" * 60)
for m in result["messages"]:
    m.pretty_print()