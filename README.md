# GEN AI Hackathon - Team 2 

## USECASE 
- Objective: Develop a generative AI agent that provides lead time insights based on user prompts to improve supply chain efficiency and responsiveness.
- Impact: The solution aims to enhance customer satisfaction, reduce inventory costs, and streamline supply chain operations by delivering accurate lead time insights through an interactive chatbot interface.

## SOLUTION ARCHITECTURE
- The solution includes a chat interface for entering user query and displaying relevant results. The query entered by the user is passed on to an RAG Agent using Bamboo LLM which fetches the relevant information from given data and give equivalent Python Pandas Executable code to get answer DataFrame/value . This response answer along with the user query is shared to a AssistantAgent ( Autogen ) using Meta's Llama 3.1 to generate an summarised text description for response answer . The chat interface interacts with backend Python Script by cloud hosted URL endpoints and renders the change

## Setup Locally :
You will require : 
  - PANDASAI Api key , Get It here [https://www.pandabi.ai/admin/api-keys]
  - Groq API [https://console.groq.com/keys]
  - Railway Account [https://railway.app/]

1. clone/fork this repo
2. Add your enviornment variables
3. Connect Railway with your github repo
4. Update Hosting settings :
   - Build Command : pip install -r requirements.txt
   - Start Command : uvicorn api.main:app --host 0.0.0.0 --port 8000
