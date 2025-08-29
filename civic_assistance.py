import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Initialize the LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    print("Please ensure your GOOGLE_API_KEY is set correctly in the .env file.")
    exit()

# Defining Prmpt Template
prompt = PromptTemplate(
    template="""You are Nalvadi 4.0, a helpful AI assistant for the citizens of Mysuru.
    Your goal is to explain civic processes in simple, clear Kannada and English.
    Your tone should be helpful, respectful, and encouraging.
    
    Answer the following question: {question}""",
    input_variables=["question"]
)

# Creating Chain that combines prompt and LLM
civic_chain = prompt | llm

if __name__ == "__main__":
    print("Welcome to the Nalvadi 4.0 Civic Assistant (Test Script)")
    print("-" * 50)
    
    # You can change this question to test different things
    test_question = "What are the timings for the Mysuru Palace?"
    
    print(f"Asking: {test_question}\n")
    
    # The .invoke() method runs the chain and gets the result
    response = civic_chain.invoke({"question": test_question})
    
    print("AI Response:")
    print(response.content)
