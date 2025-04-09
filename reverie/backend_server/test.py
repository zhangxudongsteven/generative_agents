"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import os
from utils import *
from openai import OpenAI

# Try to get the API key from environment variable first, fall back to the one in utils.py
openai_api_key = os.environ.get("OPENAI_API_KEY", "") or openai_api_key
openai_base = os.environ.get("OPENAI_BASE_URL", "") or openai_base_url
openai_api_model = os.environ.get("OPENAI_MODEL", "") or openai_model

client = OpenAI(api_key=openai_api_key, base_url=openai_base)


def llm(prompt, gpt_parameter): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of LLM response. 
  """
  try:
    print(f"GPT Parameter: {gpt_parameter}")
    # Using the chat completions endpoint instead of completions
    response = client.chat.completions.create(
                model=openai_api_model,
                messages=[
                  {"role": "user", "content": prompt}
                ],
                temperature=0 if gpt_parameter["temperature"] is None else gpt_parameter["temperature"],
                max_tokens=1000 if gpt_parameter["max_tokens"] is None else gpt_parameter["max_tokens"],
                top_p=1 if gpt_parameter["top_p"] is None else gpt_parameter["top_p"],
                frequency_penalty=0 if gpt_parameter["frequency_penalty"] is None else gpt_parameter["frequency_penalty"],
                presence_penalty=0 if gpt_parameter["presence_penalty"] is None else gpt_parameter["presence_penalty"],
                stop=None if gpt_parameter["stop"] is None else gpt_parameter["stop"])
    print(f"LLM Response: {str(response)}")
    return response.choices[0].message.content
  except Exception as e: 
    print(f"Error in GPT_request: {e}")
    return "TOKEN LIMIT EXCEEDED"


prompt = """
---
Character 1: Maria Lopez is working on her physics degree and streaming games on Twitch to make some extra money. She visits Hobbs Cafe for studying and eating just about everyday.
Character 2: Klaus Mueller is writing a research paper on the effects of gentrification in low-income communities.

Past Context: 
138 minutes ago, Maria Lopez and Klaus Mueller were already conversing about conversing about Maria's research paper mentioned by Klaus This context takes place after that conversation.

Current Context: Maria Lopez was attending her Physics class (preparing for the next lecture) when Maria Lopez saw Klaus Mueller in the middle of working on his research paper at the library (writing the introduction).
Maria Lopez is thinking of initating a conversation with Klaus Mueller.
Current Location: library in Oak Hill College

(This is what is in Maria Lopez's head: Maria Lopez should remember to follow up with Klaus Mueller about his thoughts on her research paper. Beyond this, Maria Lopez doesn't necessarily know anything more about Klaus Mueller) 

(This is what is in Klaus Mueller's head: Klaus Mueller should remember to ask Maria Lopez about her research paper, as she found it interesting that he mentioned it. Beyond this, Klaus Mueller doesn't necessarily know anything more about Maria Lopez) 

Here is their conversation. 

Maria Lopez: "
---
Output the response to the prompt above in json. The output should be a list of list where the inner lists are in the form of ["<Name>", "<Utterance>"]. Output multiple utterances in ther conversation until the conversation comes to a natural conclusion.
Example output json:
{"output": "[["Jane Doe", "Hi!"], ["John Doe", "Hello there!"] ... ]"}
"""

gpt_parameter = {"max_tokens": 16000,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0,
                 "stop": None}

if __name__ == "__main__":
  response = llm(prompt=prompt, gpt_parameter=gpt_parameter)
  print(response)
