# ex 4
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
# from main import INVOICE_SCHEMA

load_dotenv()

openrouter_api = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=openrouter_api,
    base_url="https://openrouter.ai/api/v1",
)

NON_STRICT_SCHEMA = {
    "$defs": {
        "BaseCustomer": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"},
            },
            "required": ["name", "email"],
        }
    },
    "type": "object",
    "properties": {
        "customer": {
            "allOf": [
                {"$ref": "#/$defs/BaseCustomer"},
                {
                    "properties": {
                        "tier": {"type": "string", "enum": ["gold", "platinum"]},
                        "discount_rate": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["tier", "discount_rate"],
                }
            ]
        }
    },
    "required": ["customer"],
}

STRICT_COMPATIBLE_SCHEMA = {
    "type": "object",
    "properties": {
        "customer": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"},
                "tier": {"type": "string", "enum": ["gold", "platinum"]},
                "discount_rate": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["name", "email", "tier", "discount_rate"],
            "additionalProperties": False,
        }
    },
    "required": ["customer"],
    "additionalProperties": False,
}

INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "customer": {
            "type": "string",
            "minLength": 1,
            "maxLength": 200,
        },
        "line_items": {
            "type": "array",
            "items": {
                "oneOf" : [
                {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["product"]},
                        "sku": {"type": "string", "pattern": "^[A-Z0-9-]+$"},
                        "qty": {"type": "integer", "minimum": 1},
                        "unit_usd": {"type": "number", "minimum": 0},
                    },
                    "required": ["kind" , "sku", "qty", "unit_usd"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["service"]},
                        "description": {"type": "string", "minLength": 1},
                        "hours": {"type": "number", "minimum": 0},
                        "rate_per_hour": {"type": "number", "minimum": 0},
                    },
                    "required": ["kind", "description", "hours", "rate_per_hour"],
                    "additionalProperties": False,
                },
            ],
        },
        },
        "total_usd": {"type": "number", "minimum": 0},
        "currency": {"type": "string", "enum": ["USD", "EUR", "INR"]},
    },
    "required": ["customer", "line_items", "total_usd", "currency"],
    "additionalProperties": False,
}

PROBE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}

CANDIDATES = [
    "tencent/hy3:free",
    "poolside/laguna-xs-2.1:free",
    "x-ai/grok-4.3-fast:free"
]

SYSTEM_PROMPT = "Extract invoice data as JSON. If the text is not an invoice, refuse."

inputs = [
    ("A song lyric", "Roses are red, violets are blue..."),
    ("A math proof", "Pythagorean theorem"),
    ("A blank email", "just whitespace"),
    ("A poem about a cat", "a cute cat sat on a mat..."),
    ("A recipe for chocolate cake", "Chocolate cake recipe..."),
    ("A philosophical quote", "I think therefore I am"),
    ("An image description", "A red ball on a green field"),
    ("A random string of characters", "asdf1234!@#$"),
    ("A single word", "Hello"),
    ("A polite decline", "This text does not contain an invoice")
]

working_models = []

# for model in CANDIDATES:
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=[{"role" : "user" , "content" : "Hi"}],
#             response_format={"type": "json_schema", "json_schema": {
#                 "name" : "probe" , "strict" : True , "schema" : PROBE_SCHEMA
#             }},
#             max_tokens=10
#         )
#         working_models.append(model)
#         print(f"✅ {model} supports strict mode")
#     except Exception as e:
#         print(f"❌ {model} skipped: {e}")

# measure refusals
# for model in working_models:
import time
model_name = "tencent/hy3:free"
refusals=0
hallucinations=0
errors=0    
for label , input_text in inputs:
    time.sleep(3)
    try:
        response = client.chat.completions.create(
            model = model_name,
            messages=[
                {"role" : "system" , "content" : SYSTEM_PROMPT},
                {"role" : "user" , "content" : f"Input Text : {input_text}"},
            ],
            timeout=30,
            response_format={"type": "json_schema", "json_schema": {
                "name" : "invoice" , "strict" : True , "schema" : INVOICE_SCHEMA
            }}
        )
        msg = response.choices[0].message
        if msg.refusal:
            refusals += 1
            print(f"  REFUSAL: {label}")
        else:
            hallucinations += 1
            print(f"  HALLUCINATION: {label} → {msg.content[:60]}")
    except Exception as e:
        errors += 1
        print(f"  ERROR: {label} → {e}")

print(f"\n📊 {model_name}: {refusals} refusals, {hallucinations} hallucinations, {errors} errors\n")