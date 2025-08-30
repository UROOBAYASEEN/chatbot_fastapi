

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
load_dotenv()
from agents import Agent,Tool,FileSearchTool,Runner,input_guardrail,RunContextWrapper,TResponseInputItem,GuardrailFunctionOutput,InputGuardrailTripwireTriggered
from openai import OpenAI



class CheckInput(BaseModel):
    relative_input:bool
app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@input_guardrail
async def check_service_related_question(ctx:RunContextWrapper,Agennts:Agent,input:str|TResponseInputItem):
    checking_agent=Agent(
        name="Gardrails Agents",
      instructions = (
          "You are a guardrail agent. "
          "Check the user query. "
          "- If the query is a greeting (hello, hi, thanks, how are you, etc.), allow it and set vector_store_related_query=True. "
          "- If the query is about our services or products and  want user  data like working talk, allow it and set vector_store_related_query=True. "
          "- If the query is unrelated or irrelevant, set vector_store_related_query=False."
)
       ,
        output_type=CheckInput
    )
    results=await Runner.run(starting_agent=checking_agent,input=input)
    if results.final_output.relative_input:
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
        instructions="You are a helpful agent. The user's data is stored in the vector store. If the user asks any question related to that stored data or about the user, retrieve the information from the vector store and answer smartly and clearly. Always give professional and relevant answers based only on the stored data.",

        tools=[FileSearchTool(vector_store_ids=store_vector_ids,max_num_results=1)],
        input_guardrails=[check_service_related_question]
        )
    
    try:
        result=await Runner.run(starting_agent=myagent,input=item)
        return {"message": "Data added successfully!", "data":result.final_output}
    except InputGuardrailTripwireTriggered:
        return {"message": "Data added successfully!", "data":'please Ask relative quary'}