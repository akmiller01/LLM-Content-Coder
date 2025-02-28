# Remote server Ollama install
# curl -fsSL https://ollama.com/install.sh | sh
# pip install ollama python-dotenv pydantic pandas google-genai openai

import os
import sys
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
import google
from google import genai
from openai import OpenAI, OpenAIError
import ollama
from ollama import chat
from ollama import ChatResponse
from pydantic import BaseModel
from typing import Literal
from tqdm import tqdm
import time


SINGLE_SYSTEM_PROMPT = (
    "You are classifying user text. The possible classes are:\n"
    "{}\n"
    "Please respond with only valid JSON in the following format:\n"
    "{{\n"
    "    'reasoning': 'Explain your reasoning for why or why not the text is a match with a class.',\n"
    "    'classifications': 'The name of the class that best matches the user text',\n"
    "    'confidence': 'Your confidence in the classification on a scale from 0 to 100; 100 being most confident.'\n"
    "}}"
)
GEMINI_SINGLE_SYSTEM_PROMPT = SINGLE_SYSTEM_PROMPT + "\nThe text to classify follows below:\n{}"
MULTI_SYSTEM_PROMPT = (
    "You are classifying user text. The possible classes are:\n"
    "{}\n"
    "Please respond with only valid JSON in the following format:\n"
    "{{\n"
    "    'reasoning': 'Explain your reasoning for why or why not the text is a match with each class.',\n"
    "    'classifications': ['List all of the matched classes'],\n"
    "    'confidence': 'Your confidence in the classification on a scale from 0 to 100; 100 being most confident.'\n"
    "}}"
)
GEMINI_MULTI_SYSTEM_PROMPT = MULTI_SYSTEM_PROMPT + "\nThe text to classify follows below:\n{}"


def createStructuredOutputClass(classes, multi):
    if multi:
        class ReasonedClassification(BaseModel):
            reasoning: str
            classifications: list[Literal[tuple(classes)]]
            confidence: int
    else:
        class ReasonedClassification(BaseModel):
            reasoning: str
            classifications: Literal[tuple(classes)]
            confidence: int
    return ReasonedClassification


def createFormattedPromptContents(value, classes, model, multi):
    if model == 'gemini':
        if multi:
            formattedPromptContents = GEMINI_MULTI_SYSTEM_PROMPT.format(
                "\n".join(["- " + class_name for class_name in classes]),
                value
            )
        else:
            formattedPromptContents = GEMINI_SINGLE_SYSTEM_PROMPT.format(
                "\n".join(["- " + class_name for class_name in classes]),
                value
            )
    else:
        if multi:
            formattedPromptContents = [
                {
                    'role': 'system',
                    'content': MULTI_SYSTEM_PROMPT.format(
                        "\n".join(["- " + class_name for class_name in classes])
                    ),
                },
                {
                    'role': 'user',
                    'content': value,
                },
            ]
        else:
            formattedPromptContents = [
                {
                    'role': 'system',
                    'content': SINGLE_SYSTEM_PROMPT.format(
                        "\n".join(["- " + class_name for class_name in classes])
                    ),
                },
                {
                    'role': 'user',
                    'content': value,
                },
            ]
    return formattedPromptContents


def geminiClassify(client, model, value, classes, multi, structuredOutputClass):
    formattedPromptContents = createFormattedPromptContents(value, classes, model, multi)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=formattedPromptContents,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': structuredOutputClass,
                },
            )
            json_response = json.loads(response.text)
            return json_response
        except google.genai.errors.ServerError as e:
            print(f"Connection error: {e}")
            if attempt < max_retries - 1:
                sleep_duration = (2 ** attempt) * 1
                print(f"Retrying in {sleep_duration} seconds...")
                time.sleep(sleep_duration)
            else:
                print("Max retries reached.  Returning None.")
                return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response text: {response.text}")
            if attempt < max_retries - 1:
                sleep_duration = (2 ** attempt) * 1
                print(f"Retrying in {sleep_duration} seconds...")
                time.sleep(sleep_duration)
            else:
                print("Max retries reached.  Returning None.")
                return None
    return None


def gptClassify(client, model, value, classes, multi, structuredOutputClass):
    formattedPromptContents = createFormattedPromptContents(value, classes, model, multi)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.beta.chat.completions.parse(
                model='gpt-4o-mini',
                messages=formattedPromptContents,
                response_format=structuredOutputClass
            )
            json_response = completion.choices[0].message.parsed
            return json_response.model_dump()
        except OpenAIError as e:
            print(f"Connection error: {e}")
            if attempt < max_retries - 1:
                sleep_duration = (2 ** attempt) * 1
                print(f"Retrying in {sleep_duration} seconds...")
                time.sleep(sleep_duration)
            else:
                print("Max retries reached.  Returning None.")
                return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response text: {response.text}")
            if attempt < max_retries - 1:
                sleep_duration = (2 ** attempt) * 1
                print(f"Retrying in {sleep_duration} seconds...")
                time.sleep(sleep_duration)
            else:
                print("Max retries reached.  Returning None.")
                return None
    return None


def ollamaClassify(client, model, value, classes, multi, structuredOutputClass):
    formattedPromptContents = createFormattedPromptContents(value, classes, model, multi)
    response: ChatResponse = client(
        model=model,
        format=structuredOutputClass.model_json_schema(),
        messages=formattedPromptContents
    )
    parsed_response_content = json.loads(response.message.content)
    return parsed_response_content


def main():
    parser = argparse.ArgumentParser(
                    prog='Gemini classifier',
                    description='General purpose classifier utilizing the Gemini API')
    parser.add_argument('-f', '--filename', required=True)
    parser.add_argument('-i', '--index', type=int, default=1)
    parser.add_argument('-c','--classes', nargs='+', help='<Required> Set classes', required=True)
    parser.add_argument('-m', '--multi', action='store_true')
    parser.add_argument('-d', '--model', default='gemini')
    parser.add_argument('-o', '--outfilename')
    args = parser.parse_args()

    load_dotenv()
    if args.model == 'gemini':
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if GEMINI_API_KEY is None:
            print("Please provide a GEMINI_API_KEY in a .env file.")
            return
        client = genai.Client(api_key=GEMINI_API_KEY)
        classifyFunction = geminiClassify
    elif args.model == 'gpt':
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if OPENAI_API_KEY is None:
            print("Please provide an OPENAI_API_KEY in a .env file.")
            return
        client = OpenAI(api_key = OPENAI_API_KEY)
        classifyFunction = gptClassify
    else: # Assume all other models are served via ollama
        print(f"Pulling model: {args.model}")
        ollama.pull(args.model)
        client = chat
        classifyFunction = ollamaClassify


    structuredOutputClass = createStructuredOutputClass(args.classes, args.multi)
    data = pd.read_csv(args.filename)
    colnames = data.columns
    columnName = colnames[args.index - 1]
    column = data[[columnName]]
    column_values = [val[0] for val in column.values.tolist()]
    reasonings = []
    classifications = []
    confidences = []
    for value in tqdm(column_values):
        modelResponse = classifyFunction(client, args.model, value, args.classes, args.multi, structuredOutputClass)
        if modelResponse is not None:
            reasonings.append(modelResponse['reasoning'])
            if args.multi:
                classifications.append(', '.join(modelResponse['classifications']))
            else:
                classifications.append(modelResponse['classifications'])
            confidences.append(modelResponse['confidence'])
        else:
            reasonings.append("")
            classifications.append("")
            confidences.append("")
    data['llm_reasonings'] = reasonings
    data['llm_classifications'] = classifications
    data['llm_confidence'] = confidences
    if not args.outfilename:
        fileSplitEx, _ = os.path.splitext(filename)
        outfilename = fileSplitEx + "_llm_classified.csv"
    else:
        outfilename = args.outfilename
    data.to_csv(outfilename, index=False)
        

if __name__ == '__main__':
    main()