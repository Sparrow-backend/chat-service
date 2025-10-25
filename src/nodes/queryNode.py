



from datetime import datetime
from typing_extensions import Literal
from src.llms.groqllm import GroqLLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, get_buffer_string
from src.utils.prompts import clarification_with_user_instructions, transform_messages_into_customer_query_brief_prompt
from src.states.queryState import SparrowAgentState, ClarifyWithUser, CustomerQuestion
from src.utils.utils import get_today_str

class QueryNode:
    def __init__(self, llm):
        self.llm = llm
        
    def clarify_with_user(self, state: SparrowAgentState) -> SparrowAgentState:
        """
        Determine if the user's request contains sufficient information to proceed.
        Returns updated state with clarification status.
        """
        try:
            # Use structured output with method="json_mode" for better compatibility
            structured_output_model = self.llm.with_structured_output(
                ClarifyWithUser,
                method="json_mode",
                include_raw=False
            )
            
            response = structured_output_model.invoke([
                SystemMessage(
                    content="You are a helpful assistant that responds in JSON format. Route the input to yes or no based on the need of clarification of the query."
                ),
                HumanMessage(
                    content=clarification_with_user_instructions.format(
                        messages=get_buffer_string(messages=state.get("messages", [])),
                        date=get_today_str()
                    )
                )
            ])
            
            print("CLARIFICATION RESPONSE:", response)
            
            # Update state based on response
            updated_state = {**state}
            
            if response.need_clarification == 'yes':
                updated_state.update({
                    "messages": state.get("messages", []) + [AIMessage(content=response.question)],
                    "clarification_complete": False,
                    "needs_clarification": True
                })
            else:
                updated_state.update({
                    "messages": state.get("messages", []) + [AIMessage(content=response.verification)],
                    "clarification_complete": True,
                    "needs_clarification": False
                })
            
            return updated_state
            
        except Exception as e:
            print(f"Error in clarify_with_user: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # Fallback: Ask for clarification if there's an error
            return {
                **state,
                "messages": state.get("messages", []) + [
                    AIMessage(content="I'd be happy to help! Could you please provide more details about what you need? For example, if you want to track a package, please share the tracking number.")
                ],
                "clarification_complete": False,
                "needs_clarification": True,
                "error": str(e)
            }

    def write_query_brief(self, state: SparrowAgentState) -> SparrowAgentState:
        """
        Transform the conversation history into a comprehensive customer query brief.
        """
        try:
            # Use structured output with json_mode for better compatibility
            structured_output_model = self.llm.with_structured_output(
                CustomerQuestion,
                method="json_mode",
                include_raw=False
            )
            
            messages = state.get("messages", [])
            print("STATE MESSAGES:", messages)
            
            if not messages:
                print("ERROR: No messages in state")
                return {
                    **state,
                    "query_brief": "",
                    "error": "No messages available for query brief creation"
                }
            
            prompt = transform_messages_into_customer_query_brief_prompt.format(
                messages=get_buffer_string(messages),
                date=get_today_str()
            )
            print("PROMPT:", prompt[:200], "...")  # Print first 200 chars only
            
            # Get structured response
            response = structured_output_model.invoke([
                SystemMessage(content="You are a helpful assistant that responds in JSON format."),
                HumanMessage(content=prompt)
            ])
            print("STRUCTURED RESPONSE:", response)
            
            if response is None:
                print("ERROR: Structured response is None")
                return {
                    **state,
                    "query_brief": "",
                    "error": "Failed to generate structured response"
                }
            
            return {
                **state,
                "query_brief": response.query_brief,
                "master_messages": [HumanMessage(content=response.query_brief)],
                "query_brief_complete": True
            }
            
        except Exception as e:
            print(f"Error in write_query_brief: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # Fallback: Create a simple query brief from the messages
            messages = state.get("messages", [])
            if messages:
                # Extract the last user message as the query brief
                user_messages = [msg.content for msg in messages if hasattr(msg, 'type') and msg.type == 'human']
                fallback_brief = user_messages[-1] if user_messages else "Help with parcel query"
            else:
                fallback_brief = "Help with parcel query"
            
            return {
                **state,
                "query_brief": fallback_brief,
                "master_messages": [HumanMessage(content=fallback_brief)],
                "query_brief_complete": True,
                "error": str(e)
            }
