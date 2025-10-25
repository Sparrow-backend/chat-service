from typing_extensions import TypedDict, Annotated, Sequence, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool 
from pydantic import BaseModel, Field
import operator 

class ExecutorState(TypedDict):
    """
    State for the executor agent containing message history and research metadata.
    """
    executor_messages: Annotated[Sequence[BaseMessage], add_messages]
    execution_job: str
    executor_data: List[str]

class ExecutorOutputState(TypedDict):
    """
    Output state for the executor agent containing final executor results.
    """
    output: str
    executor_data: List[str]
    executor_messages: Annotated[Sequence[BaseMessage], add_messages]


class PlannerOutput(BaseModel):
    """Simplified output for the planner that only returns execution jobs"""
    executor_jobs: List[str] = Field(description="List of execution jobs to be completed")

class MasterState(TypedDict):
    """Master orchestrator state"""
    query_brief: str
    execution_jobs: List[str]
    completed_jobs: Annotated[List[str], operator.add]
    worker_outputs: Annotated[List[ExecutorOutputState], operator.add]
    final_output: str


