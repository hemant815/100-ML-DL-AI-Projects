from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import re
import torch


app = FastAPI(title='Text Summarizer App', description='text summrizer using t5 transformer', version='1.0')

model = T5ForConditionalGeneration.from_pretrained('./data')
tokenizer = T5Tokenizer.from_pretrained('./data')

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
model.to(device)

template = Jinja2Templates(directory='.')

#input schema for dialogue ==> str
class DialogueInput(BaseModel):
    dialogue:str


def clean_data(text):
    text = re.sub (r"\r\n"," ", text) # Lines
    text = re.sub(r"\s+",' ', text) # spaces
    text = re.sub(r"<.*?>", " " , text) # html tags <p> <h1>
    text = text.strip().lower()
    return text


def summarize_dialogue(dialogue : str) -> str:
    dialogue = clean_data(dialogue) # clean

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # generate the summary => token ids
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    
    # decoded our output
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
    return summary


# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return template.TemplateResponse("index.html", {"request": request})

