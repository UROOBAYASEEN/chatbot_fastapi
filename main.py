

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
load_dotenv()
from agents import Agent,Tool,FileSearchTool,Runner,input_guardrail,RunContextWrapper,TResponseInputItem,GuardrailFunctionOutput,InputGuardrailTripwireTriggered
from openai import OpenAI


texts=""
class CheckInput(BaseModel):
    relative_input_service_related_question:bool
app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@input_guardrail
async def check_service_related_question(ctx:RunContextWrapper,Agennts:Agent,input:str|TResponseInputItem):
    checking_agent=Agent(
        name="Gardrails Agents",
      instructions ="You are a guardrail agent for a website AI chatbot. Your job is to take the user’s input and decide whether the query is related to the website’s services or information.Set the variable relative_input_service_related_question = True if the user:Sends a greeting (hello, hi, thanks, how are you)Asks about the website’s services, workshops, products, pricing, or contact informationSet the variable vector_store_related_query = False if the query is irrelevant or unrelated to the website services, such as general knowledge questions, personal topics, or unrelated subjects.Do not provide answers. Only classify whether the query should be allowed or blocked based on its relevance."
      ,
        output_type=CheckInput
    )
    results=await Runner.run(starting_agent=checking_agent,input=input)
    if results.final_output.relative_input_service_related_question:
        return GuardrailFunctionOutput(output_info="valid data",tripwire_triggered=False)
    else:
        return GuardrailFunctionOutput(output_info="invalid data",tripwire_triggered=True)


@app.get("/")
def home():
    return {"message": "FastAPI is running!"}



@app.post("/chatbot")
async def add_data(item:str):
   
   
    store_vector_ids=[]
    vector_stores = client.vector_stores.list()

# Print details
    for vs in vector_stores.data:
        store_vector_ids.append(vs.id)
    
    # print(store_vector_ids)
    myagent=Agent(
        name="helpfull asistance",
        instructions="You are an AI chatbot. When the customer asks a question, retrieve  our relevant information from the vector store containing their website data, services, contact information, pricing, and address. Provide clear, professional, and logical answers that are on-point and relevant to the user’s query. Additionally, analyze the user’s input to understand how you can help them and suggest useful next steps, without giving overly long or detailed responses.",
        tools=[FileSearchTool(vector_store_ids=store_vector_ids,max_num_results=1)],
        input_guardrails=[check_service_related_question]
        )
    
    try:
        result=await Runner.run(starting_agent=myagent,input=item)
        return {"message": "Data added successfully!", "data":result.final_output}
    except InputGuardrailTripwireTriggered:
        return {"message": "Data added successfully!", "data":'please Ask relative quary'}