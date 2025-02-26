import os
import sys
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from typing import Literal
from tqdm import tqdm


SINGLE_SYSTEM_PROMPT = (
    "You are classifying user text. The possible classes are:\n"
    "{}\n"
    "Please respond with only valid JSON in the following format:\n"
    "{{\n"
    "    'reasoning': 'Explain your reasoning for why or why not the text is a match with a class.',\n"
    "    'classifications': 'The name of the class that best matches the user text',\n"
    "    'confidence': 'Your confidence in the classification on a scale from 0 to 100; 100 being most confident.'\n"
    "}}\n"
    "The text to classify follows below:\n"
    "{}"
)
MULTI_SYSTEM_PROMPT = (
    "You are classifying user text. The possible classes are:\n"
    "{}\n"
    "Please respond with only valid JSON in the following format:\n"
    "{{\n"
    "    'reasoning': 'Explain your reasoning for why or why not the text is a match with each class.',\n"
    "    'classifications': ['List all of the matched classes'],\n"
    "    'confidence': 'Your confidence in the classification on a scale from 0 to 100; 100 being most confident.'\n"
    "}}\n"
    "The text to classify follows below:\n"
    "{}"
)


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


def createFormattedPromptContents(value, classes, multi):
    if multi:
        formattedPromptContents = MULTI_SYSTEM_PROMPT.format(
            "\n".join(["- " + class_name for class_name in classes]),
            value
        )
    else:
        formattedPromptContents = SINGLE_SYSTEM_PROMPT.format(
            "\n".join(["- " + class_name for class_name in classes]),
            value
        )
    return formattedPromptContents


def geminiClassify(client, value, classes, multi, structuredOutputClass):
    formattedPromptContents = createFormattedPromptContents(value, classes, multi)
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


def main():
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY is None:
        print("Please provide a GEMINI_API_KEY in a .env file.")
        return
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    parser = argparse.ArgumentParser(
                    prog='Gemini classifier',
                    description='General purpose classifier utilizing the Gemini API')
    parser.add_argument('-f', '--filename', required=True)
    parser.add_argument('-i', '--index', type=int, default=1)
    parser.add_argument('-c','--classes', nargs='+', help='<Required> Set classes', required=True)
    parser.add_argument('-m', '--multi', action='store_true')
    parser.add_argument('-o', '--outfilename')
    args = parser.parse_args()

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
        geminiResponse = geminiClassify(client, value, args.classes, args.multi, structuredOutputClass)
        reasonings.append(geminiResponse['reasoning'])
        if args.multi:
            classifications.append(', '.join(geminiResponse['classifications']))
        else:
            classifications.append(geminiResponse['classifications'])
        confidences.append(geminiResponse['confidence'])
    data['gemini_reasonings'] = reasonings
    data['gemini_classifications'] = classifications
    data['gemini_confidence'] = confidences
    if not args.outfilename:
        fileSplitEx, _ = os.path.splitext(filename)
        outfilename = fileSplitEx + "_gemini_classified.csv"
    else:
        outfilename = args.outfilename
    data.to_csv(outfilename, index=False)
        

if __name__ == '__main__':
    main()