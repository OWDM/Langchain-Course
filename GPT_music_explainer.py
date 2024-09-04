import os
import streamlit as st 


from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain 



# Set OpenAI API key (ensure this is managed securely in production)
apikey = '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'
os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title("GPT Music Explainer")
prompt = st.text_input('Plug in your prompt here')




# Define the prompt template
title_template = PromptTemplate(
    input_variables=['topic'],
    template='explain this song line by line: {topic} do not repeat lines that you have already explained.'
)

# Define the LLM with the correct class for chat models model="gpt-3.5-turbo",

llm = ChatOpenAI( model="gpt-3.5-turbo",temperature=0.9)
title_chain = LLMChain(llm=llm, prompt = title_template)

# Run the chain if the user provides input
if prompt:
    response = title_chain.run(topic=prompt)
    st.write(response)