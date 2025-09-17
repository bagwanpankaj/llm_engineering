## Use Python

python3.12

### Create an environment (One Time)

```
python3.12 -m venv venv
```

### Install requirements
```
python3.12 -m pip install -r requirements.txt
```

### Launch  Jupyter Lab
```
jupyter lab
```

## Parts of LLM Development

### Models

- Open Source/Closed Source Models
- Multi-Modal
- Architecture
- Selecting

### Tools

- HuggingFace
- LangChain
- Gradio
- Weights & Biases
- Modal

### Techniques

- APIs
- Multi-shot prompting
- RAG
- Fine Tuning
- Agentization

### Closed Source Models

- GPT from OpenAI
- Claude from Anthropic
- Gemini from Google (uses OLLAMA)
- Command R from Cohere
- Perplexity (Search engine)

### Open Source Frontier Models

- Llama from Meta
- Mixtral from Mistral
- Qwen from Alibaba Cloud
- Gemma from Google
- Phi from Microsoft

### Ways to use Models

- Chat Interface
- Cloud APIs/LLM API (LangChain Framework, Amazon Bedrocks, Google Vertex, Azure ML)
- Direct Interface (With HuggingFace Transformers library, with Ollama to run locally)

## OL LAMA

- Get ollama from ollama.com and Install
- Start ollama server by using `ollama serve`
- Browse http://localhost:11434/

### Connect Using Python

**POST REQUEST** `http://localhost:11434/api/chat`
```
messages = [
  {"role": "user", "content": "Describe some business impact of Generative AI"}
]
OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL="llama3.2"
import ollama
response = ollama.chat(model=MODEL, message=message)
print(response['message']['content'])
```

### Transformers

1.) Introduced in 2017 by Google Scientist
2.) GPT-1 2018
3.) GPT-2 2019
4.) GPT-4 2020
5.) RLFH and ChatGPT 2022 (RLFH - Reinforcement learning from Human Feedback)
6.) GPT-4 2023
7.) GPT-4o 2024

### Tokenizers

https://platform.openai.com/tokenizer

Try to write a sentence and see how GPT converts them into token. Less common words or invented words are broken into (i.e masterers is broken into master and ers)

**Thumb Rule for typical english**

1 Token ~ 4 chars
1 Token ~ 0.75 words
1000 Token ~ 750 words

For numbers 1 Token ~ 3 digits

**Context Window**

Max number of token that a model can consider when generating next token. It includes original input prompt, subsequent conversation, latest input prompt and almost all output prompt

**NOTE:** vellum.ai publishes leader board for AI

100283767432
j@DyHqpL68

## Building AI Powered Brochure Generator

**Jupyter Lab** Run `Shift+Enter` to run commands in a cell

```
def get_links(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(website)}
      ],
        response_format={"type": "json_object"} // tells openai to respond in JSON
    )
    result = response.choices[0].message.content
    return json.loads(result)
```

### Stream chat completion (supported by GPT)

use `stream=True` as parameters to `openai.chat.completions.create` and then we need to iterate over the response we got from `openai.chat.completions.create` and use `chunk.choices[0].delta.content`. Here `delta.content` provides delta response


## Other AI Platform

```
import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from IPython.display import Markdown, display, update_display
import google.generativeai

claude = anthropic.Anthropic()

google.generativeai.configure()
```

**NOTE:** GPT allows setting `temprature` in API calls to tell engine level of creativity. It ranges between 0 to 1, where 0 being lowest creativity and 1 being highest. Example use

```
system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

prompts = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]

completion = openai.chat.completions.create(
    model='gpt-4.1-mini',
    messages=prompts,
    temperature=0.7
)
print(completion.choices[0].message.content)
```

Claude Sonnet API call

```
# Claude 4.0 Sonnet
# API needs system message provided separately from user prompt
# Also adding max_tokens

message = claude.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[
        {"role": "user", "content": user_prompt},
    ],
)

print(message.content[0].text)
```

Claude Sonnet API Call with streaming

```
# Claude 4.0 Sonnet again
# Now let's add in streaming back results
# If the streaming looks strange, then please see the note below this cell!

result = claude.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[
        {"role": "user", "content": user_prompt},
    ],
)

with result as stream:
    for text in stream.text_stream:
            print(text, end="", flush=True)
```

#### A rare problem with Claude streaming on some Windows boxes

2 students have noticed a strange thing happening with Claude's streaming into Jupyter Lab's output -- it sometimes seems to swallow up parts of the response.

To fix this, replace the code:

`print(text, end="", flush=True)`

with this:

`clean_text = text.replace("\n", " ").replace("\r", " ")`  
`print(clean_text, end="", flush=True)`

And it should work fine!

#### Google AI

```
# The API for Gemini has a slightly different structure.
# I've heard that on some PCs, this Gemini code causes the Kernel to crash.
# If that happens to you, please skip this cell and use the next cell instead - an alternative approach.

gemini = google.generativeai.GenerativeModel(
    model_name='gemini-2.0-flash',
    system_instruction=system_message
)
response = gemini.generate_content(user_prompt)
print(response.text)
```

```
# As an alternative way to use Gemini that bypasses Google's python API library,
# Google released endpoints that means you can use Gemini via the client libraries for OpenAI!
# We're also trying Gemini's latest reasoning/thinking model

gemini_via_openai_client = OpenAI(
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = gemini_via_openai_client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=prompts
)
print(response.choices[0].message.content)
```
