# Sparrow Agent Chatbot - Parcel Consolidation And Tracking Agentic AI System

## Overview
Welcome to Sparrow, a cutting-edge agentic AI chatbot engineered with LangGraph and powered by Groq's high-performance LLM. Designed for a parcel consolidation and tracking platform, Sparrow exemplifies scalable, intelligent automation, delivering seamless user assistance through sophisticated natural language processing and dynamic workflow orchestration. This project is a testament to modern AI engineering, blending modularity, efficiency, and extensibility.

## Key Features
- **Clarify Agent**: Analyzes and refines user queries into precise, summarized inputs for downstream agents.
- **Worker Agent**: Executes tasks with a robust toolkit including parcel tracking, user information retrieval, ETA estimation, and a "think" capability for complex reasoning (future-proof for additional tools).
- **Master Agent**: Orchestrates parallel worker tasks dynamically, synthesizing outputs into cohesive responses.
- **Final Graph**: Integrates the entire workflow into a streamlined, end-to-end solution.

## Technical Highlights
- **Framework & Libraries**: Built with Python, Flask, LangChain, and GroqLLM for state-of-the-art AI performance.
- **Graph Technology**: Leverages LangGraph for a modular, graph-based agent architecture.
- **Asynchronous Processing**: Utilizes async programming for high-efficiency task handling.
- **Code Quality**: Maintains a clean, well-documented codebase with modular file structures (states, nodes, graph builder).
- **Scalability**: Designed to handle increasing loads with dynamic worker allocation.

## System Architecture
The Sparrow system features a sophisticated workflow, visualized through the following diagrams:

![Workflow 1 - Clarify User Query Flow](./assets/Screenshot%202025-08-26%20034929.png)
![Workflow 3 - Worker Agent Subgraph Flow](./assets/Screenshot%202025-08-26%20033447.png)
![Workflow 4 - Master Agent Subgraph Flow](./assets/Screenshot%202025-08-26%20033458.png)
![Workflow 5 - Final Graph Flow](./assets/Screenshot%202025-08-26%20033424.png)

These graphs illustrate the journey from query intake to final response, showcasing agent-tool interactions and parallel processing.

## Project Structure
The repository is organized for clarity and scalability:

```
sparrow-agent/
├── src/
│   ├── graphs/
│   │   ├── __init__.py
│   │   ├── queryGraph.py         # Query clarification logic
│   │   ├── actionGraph.py         # Worker task execution with tools
│   │   ├── masterGraph.py         # Orchestration and output synthesis
│   │   ├── finalAgentGraph.py     # End-to-end workflow integration
│   ├── states/
│   │   ├── __init__.py
│   │   ├── conversationState.py   # Manages conversation thread states
│   │   ├── taskState.py           # Tracks task execution states
│   ├── llms/
│   │   ├── __init__.py
│   │   ├── groqllm.py   # Manages conversation thread states
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── actionNode.py            # Tool execution node
│   │   ├── masterNode.py        # Response compression node
│   │   ├── queryNode.py        
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── actionState.py              # Custom logging setup
│   │   ├── masterState.py              
│   │   ├── queryState.py            
├── app.py                         # Main Flask application entry point
├── requirements.txt               # Dependency list
├── templates/
│   ├── index.html                 # Chat interface template
├── README.md                      # This file

```

## Setup & Installation
1. **Clone the Repository**: `git clone https://github.com/Nivakaran-S/sparrow-agent.git`
2. **Install Dependencies**: `pip install -r requirements.txt` (includes Flask, LangChain, LangGraph, GroqLLM, etc.)
3. **Configure Environment**: Set `export FLASK_SECRET_KEY='your-secret-key'` and API keys for Groq.
4. **Run the Application**: `python app.py` (defaults to port 5000, adjustable via `PORT` env variable).

## Usage
- **Chat Interface**: Access at `http://localhost:5000`.
- **API Endpoints**:
  - `/chat` (POST): Send messages with JSON `{ "message": "your query" }`.
  - `/new_conversation` (POST): Reset to a new thread.
  - `/health` (GET): Check server status.
- **Interaction**: Real-time responses powered by GroqLLM and agent workflows.

## Contributions
- **Enhancements**: Add new tools to `workerAgent.py` or optimize graph logic.
- **Performance**: Improve async handling or worker scalability.
- **UI/UX**: Upgrade `index.html`


## Acknowledgements
- Powered by Groq for high-speed LLM inference.
- Built with LangGraph for advanced agent orchestration.