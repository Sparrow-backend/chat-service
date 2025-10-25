import gradio as gr
import uuid
import logging
from datetime import datetime
import os

from src.graphs.finalAgentGraph import sparrowAgent
from langchain_core.messages import HumanMessage, AIMessage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_langchain_message(message):
    """Ensure a message is a proper LangChain message object"""
    if isinstance(message, (HumanMessage, AIMessage)):
        return message
    elif isinstance(message, dict):
        content = message.get('content', str(message))
        message_type = message.get('type', 'ai')
        if message_type == 'human':
            return HumanMessage(content=content)
        else:
            return AIMessage(content=content)
    elif isinstance(message, str):
        return AIMessage(content=message)
    else:
        return AIMessage(content=str(message))


def clean_messages_list(messages):
    """Clean and ensure all messages in list are proper LangChain message objects"""
    cleaned_messages = []
    for msg in messages:
        cleaned_msg = ensure_langchain_message(msg)
        cleaned_messages.append(cleaned_msg)
    return cleaned_messages


def initialize_conversation():
    """Initialize a new conversation state"""
    return {
        'thread_id': str(uuid.uuid4()),
        'messages': [],
        'notes': [],
        'query_brief': '',
        'final_message': '',
        'created_at': datetime.now(),
        'last_updated': datetime.now()
    }


def process_message(user_message, history, conversation_state):
    """
    Process user message and return response
    
    Args:
        user_message: The user's input message
        history: Gradio chat history (list of [user_msg, bot_msg] pairs)
        conversation_state: Dictionary containing conversation context
    
    Returns:
        Tuple of (empty string, updated history, updated conversation state, status message)
    """
    try:
        if not user_message or not user_message.strip():
            return "", history, conversation_state, "Please enter a message"
        
        # Initialize conversation state if None
        if conversation_state is None:
            conversation_state = initialize_conversation()
        
        thread_id = conversation_state['thread_id']
        
        # Add user message to conversation
        human_message = HumanMessage(content=user_message)
        conversation_state['messages'].append(human_message)
        conversation_state['last_updated'] = datetime.now()
        
        # Clean messages
        cleaned_messages = clean_messages_list(conversation_state['messages'])
        
        # Prepare input for sparrow agent
        sparrow_input = {
            'messages': cleaned_messages,
            'notes': conversation_state.get('notes', []),
            'query_brief': conversation_state.get('query_brief', ''),
            'final_message': conversation_state.get('final_message', '')
        }
        
        logger.info(f"[{thread_id}] Processing message: {user_message[:100]}")
        logger.info(f"[{thread_id}] Input messages count: {len(cleaned_messages)}")
        
        # Invoke the sparrow agent
        result = sparrowAgent.invoke(sparrow_input)
        
        # Extract response message
        response_message = ""
        ai_message = None
        
        if result.get('final_message'):
            response_message = result['final_message']
            ai_message = AIMessage(content=response_message)
        else:
            result_messages = clean_messages_list(result.get('messages', []))
            
            # Find last user message index
            last_user_index = -1
            for i, msg in enumerate(result_messages):
                if isinstance(msg, HumanMessage):
                    last_user_index = i
            
            # Get first AI message after last user message
            for i in range(last_user_index + 1, len(result_messages)):
                msg = result_messages[i]
                if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                    response_message = msg.content
                    ai_message = msg
                    break
        
        if not response_message:
            response_message = "I'm processing your request. Could you provide more details?"
            ai_message = AIMessage(content=response_message)
        
        # Update conversation state
        if result.get('messages'):
            conversation_state['messages'] = clean_messages_list(result['messages'])
        else:
            conversation_state['messages'].append(ai_message)
        
        # Remove consecutive duplicates
        cleaned_conversation_messages = []
        prev_content = None
        prev_type = None
        
        for msg in conversation_state['messages']:
            current_content = msg.content if hasattr(msg, 'content') else str(msg)
            current_type = type(msg).__name__
            
            if current_content != prev_content or current_type != prev_type:
                cleaned_conversation_messages.append(msg)
                prev_content = current_content
                prev_type = current_type
        
        conversation_state['messages'] = cleaned_conversation_messages
        conversation_state['notes'] = result.get('notes', conversation_state.get('notes', []))
        conversation_state['query_brief'] = result.get('query_brief', conversation_state.get('query_brief', ''))
        conversation_state['final_message'] = result.get('final_message', conversation_state.get('final_message', ''))
        conversation_state['last_updated'] = datetime.now()
        
        # Update Gradio chat history
        history.append([user_message, response_message])
        
        # Create status message
        status_info = f"Thread: {thread_id[:8]}... | Messages: {len(conversation_state['messages'])}"
        if result.get('execution_jobs'):
            status_info += f" | Executed: {', '.join(result['execution_jobs'])}"
        elif result.get('notes') and isinstance(result['notes'], list) and result['notes']:
            status_info += f" | Note: {str(result['notes'][-1])[:50]}"
        
        logger.info(f"[{thread_id}] Response generated: {response_message[:100]}")
        logger.info(f"[{thread_id}] Final messages count: {len(conversation_state['messages'])}")
        
        return "", history, conversation_state, status_info
    
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        error_msg = f"An error occurred: {str(e)}"
        history.append([user_message, error_msg])
        return "", history, conversation_state, f"Error: {str(e)}"


def clear_conversation():
    """Clear conversation and start fresh"""
    new_state = initialize_conversation()
    logger.info(f"[{new_state['thread_id']}] New conversation started")
    return [], new_state, f"New conversation started (ID: {new_state['thread_id'][:8]}...)"


def get_conversation_info(conversation_state):
    """Get current conversation information"""
    if conversation_state is None:
        return "No active conversation"
    
    info_lines = [
        f"**Thread ID:** {conversation_state['thread_id']}",
        f"**Messages:** {len(conversation_state.get('messages', []))}",
        f"**Notes:** {len(conversation_state.get('notes', []))}",
        f"**Has Query Brief:** {bool(conversation_state.get('query_brief'))}",
        f"**Has Final Message:** {bool(conversation_state.get('final_message'))}",
        f"**Created:** {conversation_state.get('created_at', 'N/A')}",
        f"**Last Updated:** {conversation_state.get('last_updated', 'N/A')}"
    ]
    
    return "\n\n".join(info_lines)


# Create Gradio interface
with gr.Blocks(title="Sparrow Agent Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦜 Sparrow Agent Chat")
    gr.Markdown("Interact with the Sparrow AI Agent. Ask questions and get intelligent responses!")
    
    # State to store conversation context
    conversation_state = gr.State(initialize_conversation())
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                show_copy_button=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    lines=2,
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("New Conversation", variant="secondary")
                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1
                )
        
        with gr.Column(scale=1):
            gr.Markdown("### Debug Info")
            info_btn = gr.Button("Show Conversation Info")
            info_display = gr.Markdown("Click button to show info")
    
    # Event handlers
    submit_btn.click(
        fn=process_message,
        inputs=[msg, chatbot, conversation_state],
        outputs=[msg, chatbot, conversation_state, status_box]
    )
    
    msg.submit(
        fn=process_message,
        inputs=[msg, chatbot, conversation_state],
        outputs=[msg, chatbot, conversation_state, status_box]
    )
    
    clear_btn.click(
        fn=clear_conversation,
        inputs=[],
        outputs=[chatbot, conversation_state, status_box]
    )
    
    info_btn.click(
        fn=get_conversation_info,
        inputs=[conversation_state],
        outputs=[info_display]
    )
    
    # Initialize status on load
    demo.load(
        fn=lambda state: f"Ready | Thread: {state['thread_id'][:8]}...",
        inputs=[conversation_state],
        outputs=[status_box]
    )


# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get('PORT', 7860)),
        share=False
    )