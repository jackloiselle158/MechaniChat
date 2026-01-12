# MechaniChat
Local AI Chatbot that assists users with diagnosing car problems


This is a project I made in my Generative AI class (CSCI-B490) at Indiana University. It is a chatbot that helps its users with identifying, diagnosing, and servicing various car problems based on a description of the problem provided. It uses a local LLM ran using Ollama (Llama 3.2 3B to be specific). It has the ability to retreive content that is relavent to the context of a prompt using a ChromaDB Database. The main structure of the application uses Langchain to allow the LLM to do things like run each prompt through a system prompt, call various tools (functions), and remember the conversation history relative to the user's current session. The application is made easily usable through a very simple streamlit interface.

The following applications / Python libraries are needed to run this application:

ChromaDB
Ollama
LLM models (huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF:latest, nomic-embed-text)
langchain
streamlit
requests


BEFORE USING THE MAIN PROJECT: Run the db.py script to set up the vector databse. Depending on how powerful your computer is, this may take some time.
