from langchain.llms import OpenAI 

api_ky = "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"



llm = OpenAI(
    openai_api_key=api_ky
)

result = llm("write a very short peom:")

print(result)