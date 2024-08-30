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

# Set your PandasAI API key
os.environ['PANDASAI_API_KEY'] = "$2a$10$pf1hEUkF90iwg/BRgETZJOwoATmU9DO4usfWSHgVUjsbVV7w8OoeC"

# Load the CSV data
csv_file_path = os.path.join(os.path.dirname(__file__), 'data.csv')
print(csv_file_path)
df = pd.read_csv(csv_file_path)
sdf = SmartDataframe(df)

# Create a FastAPI app
app = FastAPI()

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
    
    # Return the response directly
    return {
        'question': user_question,
        'response': smart_response
    }

# To run the server, use the following command:
# uvicorn main:app --reload

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)