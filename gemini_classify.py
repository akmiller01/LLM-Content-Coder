import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from typing import Literal
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are classifying user text. The possible classes are:\n"
    "{}\n"
    "Please respond in the following JSON format:\n"
    "{{\n"
    "    'reasoning': 'Explain your reasoning for why or why not the text is a match with each class.',\n"
    "    'classifications': ['List all of the matched classes']\n"
    "    'confidence': 'Your confidence in the classification on a scale from 0 to 100; 100 being most confident.'\n"
    "}}\n"
    "The text to classify follows below:\n",
    "{}"
)


def fetchFilenameFromSys():
    filename = None
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        print("Please provide an input CSV filename.")
    return filename


def fetchColumnIndexFromSys():
    columnIndex = None
    if len(sys.argv) > 2:
        columnIndex = sys.argv[2]
    else:
        print("Please provide a column index.")
    return columnIndex


def fetchClassesFromSys():
    classes = []
    if len(sys.argv) > 3:
        classes = sys.argv[3:]
    else:
        print("Please provide classes following the filename.")


def createStructuredOutputClass(classes):
    class ReasonedClassification(BaseModel):
        reasoning: str
        classifications: list[Literal[tuple(classes)]]
        confidence: int


def geminiClassify(client, value, classes, structuredOutputClass):
    formattedPromptContents = SYSTEM_PROMPT.format(
        "\n".join(["- " + class_name for class_name in classes]),
        value
    )
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
    filename = fetchFilenameFromSys()
    if filename is None:
        return
    columnIndex = fetchColumnIndexFromSys()
    if columnIndex is None:
        return
    classes = fetchClassesFromSys()
    if len(classes) == 0:
        return

    structuredOutputClass = createStructuredOutputClass(classes)
    data = pd.read_csv(filename)
    column = data[columnIndex]
    import pdb; pdb.set_trace()
    reasonings = []
    classifications = []
    for value in tqdm(column):
        geminiResponse = geminiClassify(value)
        reasonings.append(geminiResponse['reasoning'])
        classifications.append(', '.join(geminiResponse['classifications']))
    data['gemini_reasonings'] = reasonings
    data['gemini_classifications'] = classifications
    fileSplitEx, _ = os.path.splitext(filename)
    data.to_csv(fileSplitEx + "_gemini_classified.csv")
        

if __name__ == '__main__':
    main()