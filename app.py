import os
import streamlit as st
from langchain.llms import OpenAI  


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 



# Set OpenAI API key (ensure this is managed securely in production)
apikey = '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'
os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title("GPT music explainer")
prompt = st.text_input('Plug in your prompt here')

# Define prompt templates
title_template = PromptTemplate(
    input_variables=['input'],  # Changed to match input key
    template='explain this song line by line:{input} don't repeat lines that you are already explained'
)
script_template = PromptTemplate(
    input_variables=['input'],  # Changed to match input key passed by the previous chain
    template='write me a youtube video script based on this title: {input}'
)

memory = ConversationBufferMemory(input_key='input', memory_key='chat_history')


# Initialize LLM with desired parameters
llm = OpenAI(temperature=0.9)

# Create individual chains
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script',memory=memory)

# Create a sequential chain combining the title and script chains
sequential_chain = SequentialChain(
    chains=[title_chain, script_chain],
    input_variables=['input'],  # Consistent key 'input'
    output_variables=['title', 'script'],
    verbose=True
)

# Check the expected input for debugging
if prompt:
    response = sequential_chain.invoke({'input': prompt})  # Matching input key 'input'
    st.write(response['title'])
    st.write(response['script'])

    with st.expander('Message History'):
        st.info(memory.buffer)