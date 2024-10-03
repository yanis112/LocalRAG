from src.utils import StructuredImageLoader, StructuredPDFOcrerizer
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from dotenv import load_dotenv

load_dotenv()





def predict_NuExtract(text, schema_json):
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("numind/NuExtract", trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract", trust_remote_code=True)
    model.to("cuda")
    model.eval()

    # Convert schema JSON string to formatted string
    schema = json.dumps(json.loads(schema_json), indent=4)
    
    # Prepare the input for the LLM
    input_llm = "<|input|>\n### Template:\n" + schema + "\n### Text:\n" + text + "\n<|output|>\n"
    
    # Tokenize the input
    input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=4000).to("cuda")
    
    # Generate output from the model
    output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
    
    # Extract the prediction from the output
    prediction = output.split("<|output|>")[1].split("<|end-output|>")[0]
    
    return prediction


import os
from groq import Groq

def get_model_response(model_name, prompt):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
    )
    return chat_completion.choices[0].message.content




if __name__=="__main__":
    
    schema_json = """{
    "Order": {
        "client name": "",
        "purchase order code": "",
        "due date": "",
        "services": [],
        "net price of the first service": "",
        "total price including tax": ""
    }
}"""

    typical_template = {
  "Header": {
    "company": "...",
    "address": "...",
    "contact": {
      "phone": "...",
      "email": "..."
    },
    "tax_info": {
      "vat_number": "...",
      "company_registration": "..."
    },
    "bank_info": {
      "bank_name": "...",
      "iban": "...",
      "bic": "..."
    }
  },
  "Invoice": {
    "title": "FACTURE",
    "client": {
      "name": "...",
      "address": "...",
      "vat_number": "..."
    }
  },
  "References": {
    "invoice_number": "...",
    "date": "...",
    "purchase_order": "..."
  },
  "Services": {
    "services_list": []
  },
  "Table": [
    {
      "description": "...",
      "unit_price_htva": "...",
      "quantity": ...,
      "tva_percentage": "...",
      "amount_htva": "...",
      "amount_ttc": "..."
    }
  ],
  "Total_cost": {
    "total_htva": "...",
    "tva_percentage": "...",
    "tva_amount": "...",
    "total_ttc": "..."
  },
  "Payment_info": {
    "payment_method": "...",
    "iban": "...",
    "bic": "..."
  }
}



    #loader = StructuredImageLoader()
    # Load all the images in the folder test_factures, first list all the images
    
    #make a list of all the images in the folder using os.listdir
    folder = "test_factures"
    images = os.listdir(folder)
    print(images)
    
    ocr=StructuredPDFOcrerizer()
    
    prompt=f'''Reconstruct the JSON structure of the following ocercized document, including tables, sections, and fields. \
    Pay attention to the fact that a same field/section can represent several lines in the ocerized document and that no field/section should be empty, and a text line next to a section title may belong to the section. Here is the typical structure for this kind of document: {str(typical_template)}  Here is the ocercized document to process:'''
    
    for k in images:
        
        print("NAME: ",k)
        
        #print("###################################################")
        content=ocr.extract_text(folder+"/"+k)
        
        #print("Content: ",content)
        
        #we recreate the json schema from unstructured text
        
        llm_answer=get_model_response("llama3-70b-8192", prompt+" DOCUMENT: "+content)
        
        #print("LLM answer ",llm_answer)
        
        print("#############################################")
        
        
        #print("Content: ",content)
        structured=predict_NuExtract(llm_answer, schema_json)
        print("Nu Extract: ",structured)
        print("###################################################")
        
       
    
    
    