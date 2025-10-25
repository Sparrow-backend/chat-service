from langgraph.graph import StateGraph, START, END
from src.nodes.masterNode import MasterOrchestrator
from src.states.masterState import MasterState
from src.llms.groqllm import GroqLLM


class MasterBuilder:
    def __init__(self, llm, ):
        self.llm = llm

    def build_master_graph(self):
        master_obj = MasterOrchestrator(self.llm)
        master_graph = StateGraph(MasterState)
        
        # Add nodes
        master_graph.add_node("orchestrator", master_obj.orchestrator)
        master_graph.add_node("worker_executor", master_obj.worker_executor)
        master_graph.add_node("synthesizer", master_obj.synthesizer)
        
        # Add edges
        master_graph.add_edge(START, "orchestrator")
        master_graph.add_conditional_edges("orchestrator", master_obj.assign_workers, ["worker_executor"])
        master_graph.add_edge("worker_executor", "synthesizer")
        master_graph.add_edge("synthesizer", END)
        
        return master_graph.compile()



# Building the graph
llm = GroqLLM().get_llm()
graph_builder = MasterBuilder(llm)
master_graph = graph_builder.build_master_graph()
print("Graph created successfully")