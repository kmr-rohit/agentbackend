# import os
# import openai
# import pandas as pd
# from pandasai import SmartDataframe
# from flask import Flask, request, jsonify

# # Set up environment variables
# os.environ['PANDASAI_API_KEY'] = "$2a$10$pf1hEUkF90iwg/BRgETZJOwoATmU9DO4usfWSHgVUjsbVV7w8OoeC"
# openai.api_key = "your_openai_api_key"

# # Load the CSV data
# df = pd.read_csv("data.csv")
# sdf = SmartDataframe(df)

# # Define a function to summarize the response using OpenAI's GPT model
# def summarize_text(text):
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=f"Summarize the following text: {text}",
#         max_tokens=50,
#         n=1,
#         stop=None,
#         temperature=0.5
#     )
#     summary = response.choices[0].text.strip()
#     return summary

# # Create a Flask app
# app = Flask(__name__)

# @app.route('/ask', methods=['POST'])
# def ask_question():
#     # Get the question from the POST request
#     user_question = request.json.get('question', '')
    
#     # Get the response from the SmartDataframe
#     smart_response = sdf.chat(user_question)
    
#     # Summarize the response
#     summary = summarize_text(smart_response)
    
#     # Return the summary as JSON
#     return jsonify({
#         'question': user_question,
#         'response': smart_response,
#         'summary': summary
#     })

# if __name__ == "__main__":
#     app.run(debug=True)


# part - 2 
# import os
# import pandas as pd
# from flask import Flask, request, jsonify
# from pandasai import SmartDataframe
# from transformers import pipeline

# os.environ['PANDASAI_API_KEY'] = "$2a$10$pf1hEUkF90iwg/BRgETZJOwoATmU9DO4usfWSHgVUjsbVV7w8OoeC"
# # Load the CSV data
# df = pd.read_csv("data.csv")
# sdf = SmartDataframe(df)

# # Set up the summarization pipeline using a free LLM from Hugging Face
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# # Create a Flask app
# app = Flask(__name__)

# @app.route('/ask', methods=['POST'])
# def ask_question():
#     # Get the question from the POST request
#     user_question = request.json.get('question', '')
    
#     # Get the response from the SmartDataframe
#     smart_response = sdf.chat(user_question)
    
#     # Summarize the response
#     summary = summarizer(smart_response, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
    
#     # Return the summary as JSON
#     return jsonify({
#         'question': user_question,
#         'response': smart_response,
#         'summary': summary
#     })

# if __name__ == "__main__":
#     app.run(debug=True)


# import os
# import pandas as pd
# import requests
# from flask import Flask, request, jsonify
# from pandasai import SmartDataframe
# os.environ['PANDASAI_API_KEY'] = "$2a$10$pf1hEUkF90iwg/BRgETZJOwoATmU9DO4usfWSHgVUjsbVV7w8OoeC"
# # Load the CSV data
# df = pd.read_csv("data.csv")
# sdf = SmartDataframe(df)

# # Set your Together AI API key
# together_ai_api_key = "85cec027fc99344316c6aac0d9e71d724adc394459e6e789b8d746c6f0d0f4a7"

# # Create a function to summarize text using Together AI
# def summarize_text_with_together_ai(text):
#     headers = {
#         'Authorization': f'Bearer {together_ai_api_key}',
#         'Content-Type': 'application/json',
#     }
#     data = {
#         "model": "together/gpt-neox-20b",
#         "prompt": f"Summarize the following text:\n\n{text}",
#         "max_tokens": 100,
#         "temperature": 0.5,
#     }

#     response = requests.post('https://api.together.xyz/generate', headers=headers, json=data)
#     response_json = response.json()

#     if response.status_code == 200:
#         summary = response_json['choices'][0]['text'].strip()
#         return summary
#     else:
#         raise Exception(f"Error from Together AI: {response_json}")

# # Create a Flask app
# app = Flask(__name__)

# @app.route('/ask', methods=['POST'])
# def ask_question():
#     # Get the question from the POST request
#     user_question = request.json.get('question', '')
    
#     # Get the response from the SmartDataframe
#     smart_response = sdf.chat(user_question)
    
#     # Summarize the response using Together AI
#     summary = summarize_text_with_together_ai(smart_response)
    
#     # Return the summary as JSON
#     return jsonify({
#         'question': user_question,
#         'response': smart_response,
#         'summary': summary
#     })

# if __name__ == "__main__":
#     app.run(debug=True)




# import os
# import pandas as pd
# import requests
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from pandasai import SmartDataframe
# os.environ['PANDASAI_API_KEY'] = "$2a$10$pf1hEUkF90iwg/BRgETZJOwoATmU9DO4usfWSHgVUjsbVV7w8OoeC"
# # Load the CSV data
# df = pd.read_csv("data.csv")
# sdf = SmartDataframe(df)

# # Set your Together AI API key
# together_ai_api_key = "85cec027fc99344316c6aac0d9e71d724adc394459e6e789b8d746c6f0d0f4a7"

# # Create a FastAPI app
# app = FastAPI()

# # Define a model for the request body
# class QuestionRequest(BaseModel):
#     question: str

# # Function to summarize text using Together AI
# def summarize_text_with_together_ai(text):
#     headers = {
#         'Authorization': f'Bearer {together_ai_api_key}',
#         'Content-Type': 'application/json',
#     }
#     data = {
#         "model": "together/gpt-neox-20b",
#         "prompt": f"Summarize the following text:\n\n{text}",
#         "max_tokens": 100,
#         "temperature": 0.5,
#     }

#     response = requests.post('https://api.together.xyz/generate', headers=headers, json=data)
#     print(response)
#     response_json = response.json()

#     if response.status_code == 200:
#         summary = response_json['choices'][0]['text'].strip()
#         return summary
#     else:
#         raise HTTPException(status_code=500, detail=f"Error from Together AI: {response_json}")

# # Define an endpoint to handle the user's question
# @app.post("/ask")
# async def ask_question(request: QuestionRequest):
#     # Get the question from the POST request
#     user_question = request.question
    
#     # Get the response from the SmartDataframe
#     smart_response = sdf.chat(user_question)
    
#     # Summarize the response using Together AI
#     summary = summarize_text_with_together_ai(smart_response)
    
#     # Return the summary
#     return {
#         'question': user_question,
#         'response': smart_response,
#         'summary': summary
#     }

# # To run the server: use the command below
# # uvicorn main:app --reload
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pandasai import SmartDataframe
from fastapi.middleware.cors import CORSMiddleware
from together import Together

# Set your PandasAI and Together API keys
os.environ['PANDASAI_API_KEY'] = "$2a$10$pf1hEUkF90iwg/BRgETZJOwoATmU9DO4usfWSHgVUjsbVV7w8OoeC"
os.environ['TOGETHER_API_KEY'] = "85cec027fc99344316c6aac0d9e71d724adc394459e6e789b8d746c6f0d0f4a7"

# Load the CSV data
csv_file_path = os.path.join(os.path.dirname(__file__), 'data.csv')
df = pd.read_csv(csv_file_path)
sdf = SmartDataframe(df)

# Create a FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all HTTP headers
)

# Define a model for the request body
class QuestionRequest(BaseModel):
    question: str

# Define an endpoint to handle the user's question
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    # Get the question from the POST request
    user_question = request.question
    
    # Get the response from the SmartDataframe
    smart_response = sdf.chat(user_question)
    
    # Initialize the list of strings
    response_list = []

    # Check if the response is a DataFrame
    if isinstance(smart_response, pd.DataFrame):
        # Convert DataFrame to a list of strings
        response_list = smart_response.apply(lambda row: ', '.join(row.values.astype(str)), axis=1).tolist()
    else:
        # If the response is not a DataFrame, ensure it's a list of strings
        if isinstance(smart_response, str):
            response_list = [smart_response]
        elif isinstance(smart_response, list):
            response_list = [str(item) for item in smart_response]
        else:
            response_list = [str(smart_response)]
    
    # Combine the question and the response list into a single text
    combined_response = {
        "question": user_question,
        "response": response_list
    }
    
    # Use the Together API to summarize the response
    client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))
    
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a text summarizer. I am trying to interact with CSV data. I will give you a Question String and a Generated Answer String. Summarize the given response based on the user query in a readable and easy-to-understand format."
            },
            {
                "role": "user",
                "content": str(combined_response)
            }
        ],
        max_tokens=2653,
        temperature=0,
        top_p=0,
        top_k=1,
        repetition_penalty=1,
        stop=["<|eot_id|>", "<|eom_id|>"],
        stream=False
    )
    
    # Extract and return the summarized response
    summarized_response = response.choices[0].message.content
    
    return {
        'question': user_question,
        'response': summarized_response
    }

# To run the server, use the following command:
# uvicorn main:app --reload

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
