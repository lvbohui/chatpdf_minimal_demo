# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import numpy as np
import os
import json
from utils import calc_cos_similarity


openai.api_type = "azure"
openai.api_base = "https://adt-openai.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY")

# 读取要处理的文件
with open("paragraph.json", "r") as f:
    texts = json.load(f)

# 对每个段落做embeddings
# text_embeddings = {}
# for text_name, text_content in texts.items():
#     response = openai.Embedding.create(
#         input=text_content,
#         engine="text-embedding-ada-002"
#     )
#     text_embeddings[text_name] = response['data'][0]['embedding']

with open("text_embeddings.json", "r") as f1:
    text_embeddings = json.load(f1)


# 对用户问题做embeddings
# user_query = "controlnet的原理是什么"
# user_query = "有哪些优化训练的方法"
user_query = "文中提到的其他相关工作有哪些"
user_query_response = openai.Embedding.create(
        input=user_query,
        engine="text-embedding-ada-002"
    )
user_query_embedding = user_query_response['data'][0]['embedding']

# 计算用户问题和每个段落的相似度，取相似度最高的几个段落，和用户的问题一起送入chatgpt
related_paragraph = calc_cos_similarity(text_embeddings, user_query_embedding)
print(f"最相关的章节为:{related_paragraph}")

reference_content = ""
for item in related_paragraph:
    reference_content += texts[item]

# 送给chatgpt, 请它回答用户问题
# defining a function to create the prompt from the system message and the messages
def create_prompt(system_message, messages):
    prompt = system_message
    message_template = "\n<|im_start|>{}\n{}\n<|im_end|>"
    for message in messages:
        prompt += message_template.format(message['sender'], message['text'])
    prompt += "\n<|im_start|>assistant\n"
    return prompt

# defining the system message
system_message_template = "<|im_start|>system\n{}\n<|im_end|>"
system_message = system_message_template.format("You are an AI assistant that helps people find information.")


# creating a list of messages to track the conversation
messages = [{"sender":"user", \
             "text":f"请阅读以下内容，然后结合你的知识，用简短易懂的中文回答我的问题。\n \
                      内容:{reference_content}\n问题:{user_query}"}]

response = openai.Completion.create(
    engine="ChatGPT-0301",
    prompt= create_prompt(system_message, messages),
    temperature=0.7,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["<|im_end|>"])
print("问题: {}".format(user_query))
print("回答: {}".format(response["choices"][0]["text"]))