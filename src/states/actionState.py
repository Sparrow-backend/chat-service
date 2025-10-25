
import operator 
from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Literal


class ExecutorState(TypedDict):
    """
    State for the executor agent containing message history and research metadata.

    This state tracks the executors conversation, iteration count for limiting
    tool calls, the executor topic being
    """
    executor_messages: Annotated[Sequence[BaseMessage], add_messages]
    execution_job: str
    executor_data: List[str]

class ExecutorOutputState(TypedDict):
    """
    Output state for the executor agent containing final executor results.

    This represents the final output of the execution process with executor_data,
    executor_messages and output from the execution process.

    """
    output: str  
    executor_data: List[str]
    executor_messages: Annotated[Sequence[BaseMessage], add_messages]


# Structured output schema 
class CustomerQuestion(BaseModel):
    """Schema for customer query brief generation"""
    query_brief: str = Field(description="A customer question that will be used to guide the execution")


