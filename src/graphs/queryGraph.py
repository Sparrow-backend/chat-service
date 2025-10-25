from langgraph.graph import StateGraph, START, END 
from src.states.queryState import SparrowAgentState, SparrowInputState

from src.nodes.queryNode import QueryNode
from src.llms.groqllm import GroqLLM



class QueryGraphBuilder:
    def __init__(self, llm):
        self.llm = llm
        self.graph = StateGraph(SparrowAgentState, input_schema=SparrowInputState)
    
    def build_query_graph(self):
        """
        Build a graph for customer query inquiry

        """
        self.query_node_obj= QueryNode(self.llm)
        print(self.llm)

        self.graph.add_node("clarify_with_user", self.query_node_obj.clarify_with_user)
        self.graph.add_node("write_query_brief", self.query_node_obj.write_query_brief)

        self.graph.add_edge(START, "clarify_with_user")
        self.graph.add_edge("clarify_with_user", "write_query_brief")
        self.graph.add_edge("write_query_brief", END)

        return self.graph
    
llm = GroqLLM().get_llm()

graph_builder=QueryGraphBuilder(llm)

graph=graph_builder.build_query_graph().compile()