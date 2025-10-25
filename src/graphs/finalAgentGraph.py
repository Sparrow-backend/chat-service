# Updated Sparrow Agent with proper routing
import asyncio
import logging
from src.graphs.masterGraph import master_graph
from src.llms.groqllm import GroqLLM
from src.states.queryState import SparrowAgentState, SparrowInputState
from langgraph.graph import StateGraph, START, END
from src.states.masterState import MasterState
from langgraph.checkpoint.memory import MemorySaver
from src.nodes.queryNode import QueryNode
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

llm = GroqLLM().get_llm()
queryNode = QueryNode(llm)

def convert_sparrow_to_master(state: SparrowAgentState) -> dict:
    """Convert SparrowAgentState to master graph input format"""
    return {
        "query_brief": state.get("query_brief", ""),
        "execution_jobs": [],
        "completed_jobs": [],
        "worker_outputs": [],
        "final_output": ''
    }

def update_sparrow_from_master(sparrow_state: SparrowAgentState, master_state: dict) -> SparrowAgentState:
    """Update sparrow state with master results"""
    # Add the final result as a message and update notes
    from langchain_core.messages import AIMessage
    
    final_output = master_state.get("final_output", "")
    if final_output:
        sparrow_state["messages"] = sparrow_state.get("messages", []) + [AIMessage(content=final_output)]
        sparrow_state["final_message"] = final_output
        
    # Add execution details to notes
    execution_jobs = master_state.get("execution_jobs", [])
    completed_jobs = master_state.get("completed_jobs", [])
    
    if execution_jobs:
        sparrow_state["notes"] = sparrow_state.get("notes", []) + [f"Execution jobs: {', '.join(execution_jobs)}"]
    
    if completed_jobs:
        sparrow_state["notes"] = sparrow_state.get("notes", []) + [f"Completed: {', '.join(completed_jobs)}"]
    
    return sparrow_state

def route_after_clarification(state: SparrowAgentState) -> str:
    """Route based on clarification status from queryNode response"""
    
    # Check if clarification_complete flag is set (most reliable)
    clarification_complete = state.get("clarification_complete", False)
    needs_clarification = state.get("needs_clarification", True)
    
    if clarification_complete or not needs_clarification:
        print("Clarification complete, proceeding to query brief")
        return "write_query_brief"
    
    # Check messages for clarification status as fallback
    messages = state.get("messages", [])
    if not messages:
        return "need_clarification"
    
    # Prevent infinite clarification loops
    if len(messages) > 10:
        print("Too many clarification rounds, proceeding to query brief")
        return "write_query_brief"
    
    # Default: needs more clarification
    print("More clarification needed")
    return "need_clarification"

def route_after_query_brief(state: SparrowAgentState) -> str:
    """Route after query brief creation"""
    
    # Check if query brief exists and is adequate
    query_brief = state.get("query_brief", "")
    
    if query_brief and len(query_brief.strip()) > 20:  # Reasonable length check
        print(f"Query brief created: {query_brief[:100]}...")
        return "master_subgraph"
    else:
        # Check how many times we've tried
        messages = state.get("messages", [])
        if len(messages) > 15:  
            print("Too many attempts, ending conversation")
            return "__end__"
        
        print("Query brief insufficient or missing, going back to clarification")
        state["notes"] = state.get("notes", []) + ["Query brief creation failed, requesting more clarification"]
        return "clarify_with_user"

def need_clarification(state: SparrowAgentState) -> SparrowAgentState:
    """Handle case where clarification is needed"""
    from langchain_core.messages import AIMessage
    
    print("Additional clarification needed.")
    
    # Add a message indicating we need more information
    clarification_msg = AIMessage(
        content="I need a bit more information to help you effectively. Could you provide more details about your request?"
    )
    
    state["messages"] = state.get("messages", []) + [clarification_msg]
    state["notes"] = state.get("notes", []) + ["Requested additional clarification from user"]
    
    return state

def run_master_subgraph(state: SparrowAgentState) -> SparrowAgentState:
    """Run the master subgraph - using sync version to avoid async issues with Send"""
    try:
        print("Running master subgraph...")
        master_input = convert_sparrow_to_master(state)
        
        # Use invoke instead of ainvoke to avoid issues with Send
        master_result = master_graph.invoke(master_input)
        
        return update_sparrow_from_master(state, master_result)
        
    except Exception as e:
        logger.error(f"Master subgraph failed: {e}")
        return {**state, "error": str(e)}

def route_after_need_clarification(state: SparrowAgentState) -> str:
    """Route after need_clarification node - always end to wait for user input"""
    return "__end__"

# Build the graph
sparrowAgentBuilder = StateGraph(SparrowAgentState, input_schema=SparrowInputState)

sparrowAgentBuilder.add_node("clarify_with_user", queryNode.clarify_with_user)
sparrowAgentBuilder.add_node("need_clarification", need_clarification)
sparrowAgentBuilder.add_node("write_query_brief", queryNode.write_query_brief)
sparrowAgentBuilder.add_node("master_subgraph", run_master_subgraph)

# Edges
sparrowAgentBuilder.add_edge(START, "clarify_with_user")

sparrowAgentBuilder.add_conditional_edges(
    "clarify_with_user",
    route_after_clarification,
    {
        "need_clarification": "need_clarification",
        "write_query_brief": "write_query_brief",
        "__end__": END
    }
)

# Improved clarification flow
sparrowAgentBuilder.add_conditional_edges(
    "need_clarification",
    route_after_need_clarification,
    {
        "clarify_with_user": "clarify_with_user",
        "__end__": END
    }
)

sparrowAgentBuilder.add_conditional_edges(
    "write_query_brief",
    route_after_query_brief,
    {
        "clarify_with_user": "clarify_with_user",
        "master_subgraph": "master_subgraph",
        "__end__": END
    }
)

sparrowAgentBuilder.add_edge("master_subgraph", END)

sparrowAgent = sparrowAgentBuilder.compile()