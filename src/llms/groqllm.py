from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv

class GroqLLM:
    def __init__(self):
        load_dotenv()

    def get_llm(self):
        try:
            # Don't print API key - security risk
            # print(os.getenv("GROQ_API_KEY"))
            os.environ["GROQ_API_KEY"] = self.groq_api_key = os.getenv("GROQ_API_KEY")
            # Using llama models which have better JSON support
            llm=ChatGroq(
                api_key=self.groq_api_key, 
                model="llama-3.3-70b-versatile",  # Better JSON support
                streaming=False,
                temperature=0.1  # Lower temperature for more consistent structured output
            )
            return llm
        except Exception as e:
            raise ValueError(f"Error occurred with exception: {e}")
        
    def get_moon(self):
        try:
            # Don't print API key - security risk
            # print(os.getenv("GROQ_API_KEY"))
            os.environ["GROQ_API_KEY"] = self.groq_api_key = os.getenv("GROQ_API_KEY")
            llm=ChatGroq(
                api_key=self.groq_api_key, 
                model="moonshotai/kimi-k2-instruct", 
                streaming=False
            )
            return llm
        except Exception as e:
            raise ValueError(f"Error occurred with exception: {e}")