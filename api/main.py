
# import os
# import pandas as pd
# from fastapi import FastAPI
# from pydantic import BaseModel
# from pandasai import SmartDataframe
# from fastapi.middleware.cors import CORSMiddleware
# from together import Together

# # Set your PandasAI and Together API keys
# os.environ['PANDASAI_API_KEY'] = "$2a$10$pf1hEUkF90iwg/BRgETZJOwoATmU9DO4usfWSHgVUjsbVV7w8OoeC"
# os.environ['TOGETHER_API_KEY'] = "85cec027fc99344316c6aac0d9e71d724adc394459e6e789b8d746c6f0d0f4a7"

# # Load the CSV data
# csv_file_path = os.path.join(os.path.dirname(__file__), 'data.csv')
# df = pd.read_csv(csv_file_path)
# sdf = SmartDataframe(df)

# # Create a FastAPI app
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all HTTP methods
#     allow_headers=["*"],  # Allows all HTTP headers
# )

# # Define a model for the request body
# class QuestionRequest(BaseModel):
#     question: str

# # Define an endpoint to handle the user's question
# @app.post("/ask")
# async def ask_question(request: QuestionRequest):
#     # Get the question from the POST request
#     user_question = request.question
    
#     # Get the response from the SmartDataframe
#     smart_response = sdf.chat(user_question)
    
#     # Initialize the list of strings
#     response_list = []

#     # Check if the response is a DataFrame
#     if isinstance(smart_response, pd.DataFrame):
#         # Convert DataFrame to a list of strings
#         response_list = smart_response.apply(lambda row: ', '.join(row.values.astype(str)), axis=1).tolist()
#     else:
#         # If the response is not a DataFrame, ensure it's a list of strings
#         if isinstance(smart_response, str):
#             response_list = [smart_response]
#         elif isinstance(smart_response, list):
#             response_list = [str(item) for item in smart_response]
#         else:
#             response_list = [str(smart_response)]
    
#     # Combine the question and the response list into a single text
#     combined_response = {
#         "question": user_question,
#         "response": response_list
#     }
    
#     # Use the Together API to summarize the response
#     client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))
    
#     response = client.chat.completions.create(
#         model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a text summarizer. I am trying to interact with CSV data. I will give you a Question String and a Generated Answer String. Summarize the given response based on the user query in a readable and easy-to-understand format."
#             },
#             {
#                 "role": "user",
#                 "content": str(combined_response)
#             }
#         ],
#         max_tokens=2653,
#         temperature=0,
#         top_p=0,
#         top_k=1,
#         repetition_penalty=1,
#         stop=["<|eot_id|>", "<|eom_id|>"],
#         stream=False
#     )
    
#     # Extract and return the summarized response
#     summarized_response = response.choices[0].message.content
    
#     return {
#         'question': user_question,
#         'response': summarized_response
#     }

# # To run the server, use the following command:
# # uvicorn main:app --reload

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




# import os
# import pandas as pd
# from fastapi import FastAPI
# from pydantic import BaseModel
# from pandasai import SmartDataframe, Agent
# from fastapi.middleware.cors import CORSMiddleware
# from together import Together

# # Set your PandasAI and Together API keys
# os.environ['PANDASAI_API_KEY'] = "$2a$10$FXX7Od49oF6GWDF7pqz4peMAuPjDqLmfnK4bZ158yCH7wpy7sLWly"
# os.environ['TOGETHER_API_KEY'] = "85cec027fc99344316c6aac0d9e71d724adc394459e6e789b8d746c6f0d0f4a7"

# # Load the CSV data
# csv_file_path = os.path.join(os.path.dirname(__file__), 'data.csv')
# df = pd.read_csv(csv_file_path)
# sdf = SmartDataframe(df)

# # Initialize the Agent for handling training and advanced queries
# agent = Agent(df)

# # Create a FastAPI app
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Define models for request bodies
# class QuestionRequest(BaseModel):
#     question: str

# class TrainRequest(BaseModel):
#     query: str
#     code: str

# class BatchTrainRequest(BaseModel):
#     queries: list[str]
#     codes: list[str]

# # Endpoint to handle user's question
# @app.post("/ask")
# async def ask_question(request: QuestionRequest):
#     user_question = request.question
    
#     # Try to answer using the SmartDataframe
#     smart_response = sdf.chat(user_question)
    
#     combined_response = {
#         "question": user_question,
#         "response": smart_response,
#     }
#     print(combined_response)
#     # Use the Together API to summarize the response
#     client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))
#     response = client.chat.completions.create(
#         model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "I have queried a ai agent to get data from csv on user query , but i have a dataframe as response , i dont want to show this , can you generate text description based on this data. Dont act like any ai agent , just give me an answer to my query as a normal human being. Don't add these type of sentences : So you want to get a text description of the data instead of the actual dataframe. In this case, the AI agent has returned a count of 14 orders from the supplier site Singapore. Here's a possible text description:  , So you're looking to get a text description based on the data in the dataframe, rather than the dataframe itself. In this case, I can provide a more human-friendly summary of the data."
#             },
#             {
#                 "role": "user",
#                 "content": str(combined_response)
#             }
#         ],
#         max_tokens=2653,
#         temperature=0,
#         top_p=0,
#         top_k=1,
#         repetition_penalty=1,
#         stop=["<|eot_id|>", "<|eom_id|>"],
#         stream=False
#     )
    
#     summarized_response = response.choices[0].message.content
    
#     return {
#         'question': user_question,
#         'response': summarized_response
#     }

# # Endpoint to train the agent with a single Q&A pair
# @app.post("/train")
# async def train_agent(request: TrainRequest):
#     query = request.query
#     code = request.code
    
#     # Train the agent with the provided query and code
#     agent.train(queries=[query], codes=[code])
    
#     return {"status": "success", "message": "Agent trained successfully"}

# # Endpoint to batch train the agent with multiple Q&A pairs
# @app.post("/batch-train")
# async def batch_train_agent(request: BatchTrainRequest):
#     queries = request.queries
#     codes = request.codes
    
#     if len(queries) != len(codes):
#         return {"status": "error", "message": "The number of queries and codes must be the same"}
    
#     # Train the agent with the provided batch of queries and codes
#     agent.train(queries=queries, codes=codes)
    
#     return {"status": "success", "message": "Agent batch trained successfully"}

# # To run the server, use the following command:
# # uvicorn main:app --reload

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import os
import pandas as pd
from autogen import AssistantAgent
from fastapi import FastAPI
from pydantic import BaseModel
from pandasai import SmartDataframe, Agent
from fastapi.middleware.cors import CORSMiddleware


# Set your PandasAI API key
os.environ['PANDASAI_API_KEY'] = "$2a$10$FXX7Od49oF6GWDF7pqz4peMAuPjDqLmfnK4bZ158yCH7wpy7sLWly"
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
        "api_key": "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",
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
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    user_question = request.question
    # Try to answer using the SmartDataframe
    smart_response = sdf.chat(user_question)
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