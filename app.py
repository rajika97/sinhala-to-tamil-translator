# app.py
import os
from flask import Flask, render_template, request, jsonify
import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from peft import PeftModel, PeftConfig, LoraConfig

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None
peft_model = None

def load_model():
    """Load the translation model and tokenizer"""
    global model, tokenizer, device, peft_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model and tokenizer
    model_name = 'facebook/mbart-large-50-many-to-many-mmt'
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)

    # Load the base model first
    base_model = MBartForConditionalGeneration.from_pretrained(model_name)

    # Load PeftModel with the first adapter (Sinhala-English)
    # We need to load the config first to initialize the model properly
    peft_config_si_en = PeftConfig.from_pretrained("./models/sinhala_english")
    peft_model = PeftModel.from_pretrained(base_model, "./models/sinhala_english", adapter_name="sinhala_english")

    # Load the second adapter (English-Tamil)
    peft_model.load_adapter("./models/english_tamil", adapter_name="english_tamil")

    peft_model.to(device)
    print("Model loaded successfully!")

def translate_sinhala_to_tamil(sinhala_text):
    """Translate Sinhala text to Tamil using pivot-based approach"""
    if not sinhala_text:
        return ""

    # Step 1: Sinhala to English
    peft_model.set_adapter("sinhala_english")
    tokenizer.src_lang = "si_LK"
    tokenizer.tgt_lang = "en_XX"

    inputs = tokenizer(sinhala_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    english_output = peft_model.generate(
        **inputs,
        num_beams=5,
        max_length=50,
        early_stopping=True
    )

    english_text = tokenizer.decode(english_output[0], skip_special_tokens=True)

    # Step 2: English to Tamil
    peft_model.set_adapter("english_tamil")
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "ta_IN"

    # Ensure Tamil output by forcing the Tamil language token
    forced_bos_token_id = tokenizer.lang_code_to_id["ta_IN"]

    inputs = tokenizer(english_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    tamil_output = peft_model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        num_beams=5,
        max_length=50,
        early_stopping=True
    )

    tamil_prediction = tokenizer.decode(tamil_output[0], skip_special_tokens=True)

    return {
        "sinhala": sinhala_text,
        "english_pivot": english_text,
        "tamil": tamil_prediction
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    sinhala_text = data.get('text', '')
    try:
        result = translate_sinhala_to_tamil(sinhala_text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)