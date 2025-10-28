from langgraph.graph import StateGraph, START, END
from src.states.masterState import ExecutorState, ExecutorOutputState

from src.nodes.actionNode import ExecutorNode
from src.llms.groqllm import GroqLLM
from src.utils.prompts import execution_agent_prompt, compress_execution_human_message, compress_execution_system_prompt

from src.utils.utils import think_tool, track_package, estimated_time_analysis

tools = [think_tool, track_package, estimated_time_analysis]


class ExecutorGraphBuilder:
    def __init__(self, llm ):
        self.llm = llm 
        self.graph = StateGraph(ExecutorState, output=ExecutorOutputState)
        self.tools = tools
        self.execution_agent_prompt = execution_agent_prompt
        self.compress_execution_system_prompt = compress_execution_system_prompt
        self.compress_execution_human_message = compress_execution_human_message
    
    def build_executor_graph(self):
        """Build a graph to build the executor"""
        self.executor_node_obj = ExecutorNode(
            self.llm, 
        )

        self.graph.add_node("llm_call", self.executor_node_obj.llm_call)
        self.graph.add_node("tool_node", self.executor_node_obj.tool_node)
        self.graph.add_node("compress_execution", self.executor_node_obj.compress_execution)

        self.graph.add_edge(START, "llm_call")
        self.graph.add_conditional_edges(
            "llm_call",
            self.executor_node_obj.guard_llm,
            {
                "tool_node": "tool_node",
                "compress_execution": "compress_execution",
            },
        )
        self.graph.add_edge("tool_node", "llm_call")
        self.graph.add_edge("compress_execution", END)

        return self.graph
    
    def setup_graph(self):
        return self.graph.compile()


llm=GroqLLM().get_llm()


graph_builder=ExecutorGraphBuilder(llm)
graph=graph_builder.build_executor_graph().compile()
