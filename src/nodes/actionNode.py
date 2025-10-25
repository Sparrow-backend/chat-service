from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from src.utils.prompts import execution_agent_prompt, compress_execution_system_prompt, compress_execution_human_message
from src.utils.utils import think_tool, track_package, estimated_time_analysis, get_today_str
import logging

tools = [think_tool, track_package, estimated_time_analysis]
tools_by_name = {tool.name: tool for tool in tools}

class ExecutorNode:
    """
    Executor node for handling tasks:
    1. LLM reasoning
    2. Tool invocation
    3. Final compression
    """

    def __init__(self, llm):
        self.llm = llm
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.model_with_tools = llm.bind_tools(tools)
        self.MAX_ITERATIONS = 6  # Increased to allow more tool calls (including think_tool)
        self.execution_agent_prompt_template = execution_agent_prompt
        self.compress_execution_system_prompt_template = compress_execution_system_prompt
        self.compress_execution_human_message = compress_execution_human_message
        
        # Debug tool binding
        print(f"Available tools: {list(self.tools_by_name.keys())}")

    def llm_call(self, state: dict) -> dict:
        """Calls the LLM with the executor message history and returns updated state."""
        try:
            # Ensure we have the execution job in the messages
            execution_job = state.get("execution_job", "")
            existing_messages = state.get("executor_messages", [])
            print("EXECUTOR MESSAGES MESSAGES", existing_messages)
            print("EXECUTION JOB", execution_job)

                        # If no existing messages, add the execution job as initial human message
            if not existing_messages and execution_job:
                existing_messages = [HumanMessage(content=execution_job)]
            
            # Format the prompt with current date
            formatted_prompt = self.execution_agent_prompt_template.format(date=get_today_str())
            messages = [SystemMessage(content=formatted_prompt)] + existing_messages
            
            print(f"Calling LLM with {len(messages)} messages")
            print(f"Last message: {messages[-1] if messages else 'No messages'}")
            
            response = self.model_with_tools.invoke(messages)
            
            print(f"LLM Response type: {type(response)}")
            print(f"LLM Response content: {response.content[:100] if response.content else 'No content'}...")
            print(f"Tool calls in response: {getattr(response, 'tool_calls', 'No tool_calls attribute')}")

            return {
                **state,
                "executor_messages": existing_messages + [response]
            }
            
        except Exception as e:
            return {
                **state,
                "error": str(e),
                "executor_messages": state.get("executor_messages", [])
            }

    def tool_node(self, state: dict) -> dict:
        """Executes any tools requested by the LLM and appends ToolMessages."""
        try:
            executor_messages = state.get("executor_messages", [])
            if not executor_messages:
                print("No executor messages found")
                return state
                
            last_message = executor_messages[-1]
            print(f"Last message type: {type(last_message)}")
            print(f"Last message attributes: {dir(last_message)}")
            
            # Get tool calls
            tool_calls = getattr(last_message, "tool_calls", [])
            print(f"Found {len(tool_calls)} tool calls: {tool_calls}")

            if not tool_calls:
                print("No tool calls found in last message")
                return state

            tool_outputs, new_data = [], []
            
            for call in tool_calls:
                print(f"Processing tool call: {call}")
                
                tool_name = call.get("name")
                args = call.get("args", {})
                tool_id = call.get("id")
                
                print(f"Tool: {tool_name}, Args: {args}, ID: {tool_id}")
                
                if tool_name in self.tools_by_name:
                    try:
                        print(f"Invoking tool {tool_name} with args {args}")
                        result = self.tools_by_name[tool_name].invoke(args)
                        print(f"Tool {tool_name} result: {result}")
                        
                        tool_message = ToolMessage(
                            content=str(result), 
                            name=tool_name, 
                            tool_call_id=tool_id
                        )
                        tool_outputs.append(tool_message)
                        new_data.append(str(result))
                        
                    except Exception as e:
                        error_msg = f"Tool {tool_name} failed: {e}"
                        print(f"Tool error: {error_msg}")
                        tool_outputs.append(
                            ToolMessage(
                                content=error_msg, 
                                name=tool_name, 
                                tool_call_id=tool_id
                            )
                        )
                        new_data.append(error_msg)
                else:
                    error_msg = f"Tool {tool_name} not found. Available: {list(self.tools_by_name.keys())}"
                    print(error_msg)
                    tool_outputs.append(
                        ToolMessage(
                            content=error_msg, 
                            name=tool_name, 
                            tool_call_id=tool_id
                        )
                    )

            print(f"Returning {len(tool_outputs)} tool outputs")
            
            return {
                **state,
                "executor_messages": executor_messages + tool_outputs,
                "executor_data": state.get("executor_data", []) + new_data
            }
            
        except Exception as e:
            return {
                **state,
                "error": f"Tool execution failed: {str(e)}"
            }

    def compress_execution(self, state: dict) -> dict:
        """Summarizes the execution and returns final structured output."""
        try:
            execution_job = state.get("execution_job", "Complete the assigned task")
            executor_messages = state.get("executor_messages", [])
            
            # Format the system prompt with current date
            formatted_system_prompt = self.compress_execution_system_prompt_template.format(date=get_today_str())
            
            messages = [
                SystemMessage(content=formatted_system_prompt),
                *executor_messages,
                HumanMessage(content=self.compress_execution_human_message.format(
                    shipment_request=execution_job
                ))
            ]

            response = self.llm.invoke(messages)

            executor_data = [
                str(m.content) for m in executor_messages 
                if hasattr(m, 'content') and m.content
            ]

            return {
                "output": str(response.content),
                "executor_data": executor_data,
                "executor_messages": executor_messages
            }
            
        except Exception as e:
            return {
                "output": f"Execution completed with errors: {str(e)}",
                "executor_data": state.get("executor_data", []),
                "executor_messages": state.get("executor_messages", [])
            }

    def route_after_llm(self, state: dict) -> str:
        """Route: decide whether to call a tool or finalize."""
        try:
            executor_messages = state.get("executor_messages", [])
            if not executor_messages:
                return "compress_execution"
                
            last_msg = executor_messages[-1]
            has_tool_calls = bool(getattr(last_msg, "tool_calls", None))
            
            print(f"Routing decision - Has tool calls: {has_tool_calls}")
            
            return "tool_node" if has_tool_calls else "compress_execution"
        except Exception as e:
            return "compress_execution"

    def guard_llm(self, state: dict) -> str:
        """Prevent infinite loops by limiting iterations."""
        iteration_count = state.get("iteration_count", 0) + 1
        state["iteration_count"] = iteration_count
        
        print(f"Iteration count: {iteration_count}/{self.MAX_ITERATIONS}")
        
        if iteration_count > self.MAX_ITERATIONS:
            print("Max iterations reached, finalizing...")
            return "compress_execution"
            
        return self.route_after_llm(state)