import operator 
from typing_extensions import Optional, Annotated, List, Sequence, Literal, Union

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator 


class SparrowInputState(MessagesState):
    """Input state for the full agent - only contains from the user input."""
    pass 

class SparrowAgentState(MessagesState):
    """
    Main state for the full multi-agent Sparrow customer service system.

    Extends MessagesState with additional fields for Sparrow customer service coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and main workflow.

    """
    query_brief: Optional[str]
    master_messages: Annotated[Sequence[BaseMessage], add_messages]
    notes: Annotated[list[str], operator.add] = []
    final_message: str
    clarification_complete: Optional[bool]
    needs_clarification: Optional[bool]
    query_brief_complete: Optional[bool]
    execution_jobs: Optional[List[str]]
    error: Optional[str]

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decision and questions"""

    need_clarification: Union[Literal["yes", "no"], bool] = Field(
        description="Whether the user needs to be asked a clarifying question. Can be 'yes'/'no' or true/false"
    )
    question: str = Field(
        description="A question to ask the user to clarify the need",
        default=""
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information",
        default=""
    )
    
    @field_validator('need_clarification', mode='before')
    @classmethod
    def convert_bool_to_string(cls, v):
        """Convert boolean values to yes/no strings"""
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower in ['true', '1', 'yes']:
                return "yes"
            elif v_lower in ['false', '0', 'no']:
                return "no"
        return v

class CustomerQuestion(BaseModel):
    """Schema for structured customer query brief """

    query_brief: str = Field(
        description="A customer question that will be used to guide the research."
    )