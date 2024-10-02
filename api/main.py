
import os
import base64
import io
import pandas as pd
import matplotlib.pyplot as plt
from autogen.agentchat import AssistantAgent
from fastapi import FastAPI
from pydantic import BaseModel
from pandasai import SmartDataframe, Agent
from fastapi.middleware.cors import CORSMiddleware

# Set your PandasAI API key
os.environ['PANDASAI_API_KEY'] = "zzzzzzzzz"
# Load the CSV data
csv_file_path = os.path.join(os.path.dirname(__file__), 'data.csv')
df = pd.read_csv(csv_file_path)
sdf = SmartDataframe(df)
# Initialize the Agent for handling training and advanced queries
agent = Agent(df)
# Create the LLM config for Autogen
config_list = [
    {
        "model": "llama3-8b-8192",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": "zzzzzz",
        "api_type": "groq"
    }
]
llm_config = {"config_list": config_list, "timeout": 60, "temperature": 0}
# Initialize the AssistantAgent
assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
    description="A helpful assistant"
)
# Create a FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define models for request bodies
class QuestionRequest(BaseModel):
    question: str
class TrainRequest(BaseModel):
    query: str
    code: str
class BatchTrainRequest(BaseModel):
    queries: list[str]
    codes: list[str]
# Endpoint to handle user's question

def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    user_question = request.question
    # Try to answer using the SmartDataframe
    smart_response = sdf.chat(user_question)
    # print(type(smart_response));
    if((type(smart_response) != str) & (type(smart_response) != int)): 
        combined_response = {
            "question": user_question,
            "response": smart_response,
        }
        # Use the Autogen agent to summarize the response
        response = assistant.generate_reply(
            messages=[
                {
                    "role": "system",
                    "content": "You are a text summarizer. I am trying to interact with CSV data. I will give you a Question String and a Generated Answer String. Summarize the given response based on the user query in a readable and easy-to-understand format. Dont add this statement in answer : Based on the query, it appears that you are looking for the number of orders that have a variance percentage greater than 100% and are from a supplier named StarElectronics.\n\nHere is a summary of the response:\n\n*"
                },
                {
                    "role": "user",
                    "content": str(combined_response)
                }
            ]
        )
        print(response)
        #if 'choices' in response:
        #summarized_response = response['choices'][0]['message']['content']
        #else:
            #summarized_response = "Unexpected response structure"

        #summarized_response = response['choices'][0]['message']['content']
        summarized_response=response['content']
        return {
            'question': user_question,
            'responseType':'text',
            'response': summarized_response
        }
    else:
        smart_response = str(smart_response);
        if( (type(smart_response) == str) & (smart_response.endswith('.png'))):
            image_path=smart_response
            image_base64 = convert_image_to_base64(image_path)
            return {
                'question': user_question,
                'responseType':'image',
                'response': image_base64
            }
        else : 
            combined_response = {
            "question": user_question,
            "response": smart_response,
            }
            # Use the Autogen agent to summarize the response
            response = assistant.generate_reply(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a text summarizer. I am trying to interact with CSV data. I will give you a Question String and a Generated Answer String. Summarize the given response based on the user query in a readable and easy-to-understand format. Dont add this statement in answer : Based on the query, it appears that you are looking for the number of orders that have a variance percentage greater than 100% and are from a supplier named StarElectronics.\n\nHere is a summary of the response:\n\n*"
                    },
                    {
                        "role": "user",
                        "content": str(combined_response)
                    }
                ]
            )
            print(response)
            #if 'choices' in response:
            #summarized_response = response['choices'][0]['message']['content']
            #else:
                #summarized_response = "Unexpected response structure"

            #summarized_response = response['choices'][0]['message']['content']
            summarized_response=response['content']
            return {
                'question': user_question,
                'responseType':'text',
                'response': summarized_response
            }
        
# Endpoint to train the agent with a single Q&A pair
@app.post("/train")
async def train_agent(request: TrainRequest):
    query = request.query
    code = request.code
    # Train the agent with the provided query and code
    agent.train(queries=[query], codes=[code])
    return {"status": "success", "message": "Agent trained successfully"}
# Endpoint to batch train the agent with multiple Q&A pairs
@app.post("/batch-train")
async def batch_train_agent(request: BatchTrainRequest):
    queries = request.queries
    codes = request.codes
    if len(queries) != len(codes):
        return {"status": "error", "message": "The number of queries and codes must be the same"}
    # Train the agent with the provided batch of queries and codes
    agent.train(queries=queries, codes=codes)
    return {"status": "success", "message": "Agent batch trained successfully"}
# To run the server, use the following command:
# uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
