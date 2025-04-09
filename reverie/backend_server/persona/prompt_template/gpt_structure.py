"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import random
import time
import os
import sys

from utils import *

# Try to get the API key from environment variable first, fall back to the one in utils.py
openai_api_key = os.environ.get("OPENAI_API_KEY", "") or openai_api_key
openai_base = os.environ.get("OPENAI_BASE_URL", "") or openai_base_url
openai_api_model = os.environ.get("OPENAI_MODEL", "") or openai_model

# Check if API key is provided
if not openai_api_key:
    print("ERROR: OpenAI API key is not set. Please set your OPENAI_API_KEY environment variable or update the key in utils.py")
    print("Example: export OPENAI_API_KEY='your-api-key-here'")
    sys.exit(1)

# Using the new OpenAI client library
from openai import OpenAI

# Configure OpenAI client
client = OpenAI(api_key=openai_api_key, base_url=openai_base)

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

def ChatGPT_single_request(prompt):
  temp_sleep()

  completion = client.chat.completions.create(
    model=openai_api_model,
    messages=[{"role": "user", "content": prompt}]
  )
  return completion.choices[0].message.content


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt):
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()

  try:
    completion = client.chat.completions.create(
    model=openai_api_model,
    messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

  except Exception as e:
    print(f"GPT4 ERROR: {e}")
    return "ChatGPT ERROR"


def ChatGPT_request(prompt):
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  try:
    completion = client.chat.completions.create(
      model=openai_api_model,
      messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

  except Exception as e:
    print(f"ChatGPT ERROR: {e}")
    return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat):

    try:
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except:
      pass

  return False


def ChatGPT_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat):

    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      # print ("---ashdfaf")
      # print (curr_gpt_response)
      # print ("000asdfhia")

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except:
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  if verbose:
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat):
    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose:
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except:
      pass
  print ("FAIL SAFE TRIGGERED")
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter):
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
  temp_sleep()

  try:
    print(f"GPT Request: {prompt}")
    response = client.chat.completions.create(
      model=openai_api_model,
      messages=[{"role": "user", "content": prompt}],
      temperature=0 if gpt_parameter["temperature"] is None else gpt_parameter["temperature"],
      max_tokens=1000 if gpt_parameter["max_tokens"] is None else gpt_parameter["max_tokens"],
      top_p=1 if gpt_parameter["top_p"] is None else gpt_parameter["top_p"],
      frequency_penalty=0 if gpt_parameter["frequency_penalty"] is None else gpt_parameter["frequency_penalty"],
      presence_penalty=0 if gpt_parameter["presence_penalty"] is None else gpt_parameter["presence_penalty"],
      stop=None if gpt_parameter["stop"] is None else gpt_parameter["stop"]
    )
    print(f"GPT Response: {response.choices[0].message.content}")
    return response.choices[0].message.content
  except Exception as e:
    print(f"Error in GPT_request: {e}")
    return "GPT_request ERROR"


def generate_prompt(curr_input, prompt_lib_file):
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"):
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt:
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt,
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
  if verbose:
    print (prompt)

  for i in range(repeat):
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt):
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose:
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response


def get_embedding(text, model=None):
  text = text.replace("\n", " ")
  if not text:
    text = "this is blank"

  # Use Cloudflare AI credentials from utils
  cf_client = OpenAI(
    api_key=CLOUDFLARE_API_KEY,
    base_url=f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1"
  )

  # Use the model specified in utils or fall back to default
  cf_model = model or CLOUDFLARE_AI_EMBEDDING_MODEL

  try:
    response = cf_client.embeddings.create(
      model=cf_model,
      input=[text]
    )
    return response.data[0].embedding
  except Exception as e:
    print(f"Error getting embedding: {e}")
    # Return empty embedding of typical size as fallback
    return [0.0] * 1024


if __name__ == '__main__':
  gpt_parameter = {"max_tokens": 128000,
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0,
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response):
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1:
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt,
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print(output)
