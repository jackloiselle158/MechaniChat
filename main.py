# Create database

import chromadb
import ollama

client = chromadb.PersistentClient(path="./mydb/")
collection = client.get_or_create_collection(name="docs")

# Initialize model, get all imports for it
from langchain_ollama.chat_models import ChatOllama
llm = ChatOllama(model="huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF:latest", temperature=0.4)
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import streamlit as st
import requests
from langchain_core.tools import tool
import random

#function to get car seat inspection stations (from NHTSA API)
@tool
def get_seat_stations(state: str) -> str:
    """Function to get car seat inspection stations in a given state.
    A string is returned, containing a list of car seat inspection stations in the specified state.
    The input to the function is the two-letter state abbreviation.
    For example: 'CA' for California, 'TX' for Texas, etc."""
    request = requests.get(f"https://api.nhtsa.gov/CSSIStation/state/{state}")
    if request.status_code == 200:
        data = request.json()
        results = data.get("Results", [])
        if not results:
            return f"No car seat inspection stations found in {state}."
        summary = f"Car Seat Inspection Stations in {state}:\n"
        random.shuffle(results)
        for station in results[:10]:
            summary += f"\n- {station.get('Organization', 'N/A')} {station.get('AddressLine1', 'N/A')} {station.get('City', 'N/A')}, {station.get('State', 'N/A')} {station.get('Zip', 'N/A')}"
        return summary
    else:
        return " "

#function to see recalls for a vehicle (from NHTSA API)
@tool
def get_recalls(make: str, model: str, year: str) -> str:
    """Function to get recalls for a vehicle given make, model, and year. 
    The returned string is information about various recalls."""
    request = requests.get(f"https://api.nhtsa.gov/recalls/recallsByVehicle?make={make}&model={model}&modelYear={year}")
    if request.status_code == 200:
        data = request.json()
        results = data.get("results", [])
        if not results:
            return f"No recalls found for {year} {make} {model}."
        summary = f"Recalls for {year} {make} {model}:\n"
        for i, recall in enumerate(results, 1):
            summary += f"\n{i}. {recall.get('Component', 'N/A')} : {recall.get('Summary', 'No summary available')}"
        return summary

    else:
        return " "
    
#function to get crash ratings for a vehicle (from NHTSA API)
@tool
def get_crash_ratings(make: str, model: str, year: str) -> str:
    """Function to get crash ratings for a vehicle given make, model, and year. 
    A string is returned, containing crash ratings for each vehicle ID found for the vehicle."""
    r1 = requests.get(f"https://api.nhtsa.gov/SafetyRatings/modelyear/{year}/make/{make}/model/{model}")
    if r1.status_code == 200:
        data = r1.json()
        vehicles = data.get("Results", [])
        if not vehicles:
            return f"No crash ratings found for {year} {make} {model}."
        summary = f"Crash Ratings for {year} {make} {model}:\n"
        for vehicle in vehicles:
            v_id = vehicle.get("VehicleId")
            r2 = requests.get(f"https://api.nhtsa.gov/SafetyRatings/VehicleId/{v_id}")
            if r2.status_code == 200:
                crash_data = r2.json()
                res = crash_data.get("Results", [{}])[0]
                summary += f"\nVehicle ID: {v_id}"
                summary += f"\n- Overall Rating: {res.get('OverallRating', "N/A")} stars"
                summary += f"\n- Front Crash: {res.get('OverallFrontCrashRating', 'N/A')} stars"
                summary += f"\n- Side Crash: {res.get('OverallSideCrashRating', 'N/A')} stars"
                summary += f"\n- Rollover: {res.get('RolloverRating', 'N/A')} stars\n"
        return summary
    else:
        return " "

# Function to retreive documents relative to context of a prompt (From HW1)

def get_relevant_context(prompt):
    # Generate embedding for the user's prompt
    prompt_response = ollama.embed(model="nomic-embed-text", input=prompt)
    prompt_embedding = prompt_response["embeddings"]
    results = collection.query(
        query_embeddings=prompt_embedding,
        n_results=1
    )
    relevant_document = results['documents'][0][0] if results and 'documents' in results else None
    return relevant_document

# Bind tools to the LLM
tools = [get_recalls, get_crash_ratings, get_seat_stations]
llm_with_tools = llm.bind_tools(tools)

# System prompt
system_prompt = """
    You are a well-experienced mechanic. Your job is to help automotive users with identifying and solving car problems.
    CRITICAL TOOL USAGE RULES:
    - You have access to two tools: get_recalls and get_crash_ratings
    - ONLY use get_recalls when the user EXPLICITLY asks about recalls, recall information, or recall history for a specific vehicle
    - ONLY use get_crash_ratings when the user EXPLICITLY asks about crash ratings, safety ratings, or crash test scores for a specific vehicle
    - DO NOT use these tools for general automotive questions, diagnosis, or troubleshooting
    - DO NOT use these tools unless the user's question clearly and directly requests recall or safety rating information
    - If the user asks about a car problem, sound, or issue, answer based on your knowledge. DO NOT call the tools
    You should follow the following rules with your consideration of the user's questions.
    1. You absolutely MUST consider the conversation history when determining your answer. It is crucial that your answer is relavent to the context of the conversation.
    2. If a user is asking about why their car is behaving a certain way, you should give them 3 possible suggestions as to what could be wrong with the car.
    There is a chance the user could ask about something that is not a problem, so you need to consider that as well. 
    3. If the user is asking about what kinds of cars have certain problems, you should provide three cars, specified by make, model, and year have that problem the most.
    4. If the user is asking about a problem with their car and does not provide a make, model, or year, you shouldn't try to do anything and should immediately ask them for what kind of car is having the problme they are describing.
    5. If a user wants to know common problems of a specific car, you should give it as many common problems as you can come up with, but no more than 10.
    6. When the user asks for help solving an automotive problem, you should use the following paths to assess the problem. Identify which of them you can use to determine the problem:
    Path 1: Sound type - If the user describes a sound, figure out which sounds can mean what in the context of what they are saying.
    Path 2: System - If the user specifies a problem with a system in the automobile (drivetrain, brakes, etc.) identify issues with those systems.
    Path 3: Conditions - If the user provides info on when the problem occurrs (certain speed, upon startup of car, shifting into reverse, etc.), identify certain componets that could be causing problems in those scenarios.
    Use a combination of the three paths based on whether there is info present in the prompt for them to be usable.
    7. If there are no relevant documents provided, you should still answer the question to the best of your abilities.

    

    8. If the prompt has nothing to do with automotives, you should ignore these rules and provide a normal answer to the prompt. There's no need to mention anything about automotives whatsoever in this case, just simply answer the prompt as a normal AI.

    
    Now, based on the converation history and following relevant documents, answer the user's question.

    {documents}

"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)
tmpchain = prompt | llm
memory_storage = {}
def get_session_history_fn(session_id):
    if "memory_storage" not in st.session_state:
        st.session_state.memory_storage = {}
    if session_id not in st.session_state.memory_storage:
        st.session_state.memory_storage[session_id] = ChatMessageHistory()
    return st.session_state.memory_storage[session_id]
chain = RunnableWithMessageHistory(
    tmpchain,
    history_messages_key="history",
    input_messages_key="input",
    get_session_history=get_session_history_fn
)

def should_use_tools(user_input):
    recall_keywords = ['recall', 'recalls', 'safety recall', 'recalled']
    rating_keywords = ['crash rating', 'safety rating', 'crash test', 'star rating', 'nhtsa rating']
    car_seat_keywords = ['car seat', 'child seat', 'carseat', 'cssi']
    
    input_lower = user_input.lower()
    for keyword in recall_keywords + rating_keywords + car_seat_keywords:
        if keyword in input_lower:
            return True
    return False

#function to process tool calls
def process_with_calls(user_input, context):

    if should_use_tools(user_input):
        llm_to_use = llm_with_tools
    else:
        llm_to_use = llm
    tmpchain = prompt | llm_to_use
    temp_chain = RunnableWithMessageHistory(
        tmpchain,
        history_messages_key="history",
        input_messages_key="input",
        get_session_history=get_session_history_fn
    )
    response = temp_chain.invoke(
        {
            "input": user_input,
            "documents": context if context else "No relevant context found",
        },
        config={"configurable": {"session_id": "chat"}},
    )

    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_results = []
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            if tool_name == "get_recalls":
                result = get_recalls.invoke(tool_args)
            elif tool_name == "get_crash_ratings":
                result = get_crash_ratings.invoke(tool_args)
            elif tool_name == "get_seat_stations":
                result = get_seat_stations.invoke(tool_args)
            else:
                result = "Tool not recognized."
            tool_results.append(result)

        session_history = get_session_history_fn("chat")
        session_history.add_message(response)
        return tool_results[0]
    else:
        return response.content

# Streamlit interface + functioning app

print(collection.count())
st.set_page_config(layout="centered")
st.title("MechaniChat")
user_prompt = st.text_area("Ask a question:")

if st.button("Ask MechaniChat"):
    context = get_relevant_context(user_prompt)
    st.subheader("Response:")
    st.write(process_with_calls(user_prompt, context))
    st.subheader("Relavent Context:")
    st.write(context)

