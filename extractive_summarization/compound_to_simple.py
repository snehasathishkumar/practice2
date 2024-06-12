import PyPDF2
import google.generativeai as genai
from dotenv import get_key
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

def compound_to_simple():
    # extracting text from pdf
    pdf_file = PyPDF2.PdfReader("first_chapter.pdf")
    extracted_text = ""

    for i in pdf_file.pages:
        extracted_text += i.extract_text()

    #Get API Key
    api_key = get_key('../.env', key_to_get='GEMINI_API_KEY')

    #Configure API key
    genai.configure(api_key=api_key)

    #Defining model
    gen_config = genai.GenerationConfig(
        temperature=0.5,
    )

    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash-latest',
    )

    #text-splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, 
        chunk_overlap=30,
        separators=['.','\n','\n\n']
    )

    # print('Original text:')
    # print(extracted_text)

    texts = text_splitter.split_text(extracted_text)

    converted_text = ""

    for text in texts:
        response = model.generate_content(
            f"""You are a cyber security expert and an expert summarizer. The audience is a general user who wants to know about cybersecurity. Convert the given paragraphs of text into simple sentences.
            Output should contain only the transformed text.
            Example:
            Input: Pure Extortion attacks have risen over the past year and are now a component of 9% of all breaches.
            Output: Pure Extortion attacks have risen over the past year . Pure Extortion attacks are now a component of 9% of all breaches.

            
            {text}"""
        )
        converted_text += response.text

    # print('Processed text:')
    # print(converted_text)

    return converted_text