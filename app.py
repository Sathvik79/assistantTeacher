from flask import Flask, request, jsonify, render_template
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
import torch

app = Flask(__name__)

# Loading the pre-trained GPT-2 model for response generation
gpt2_model_name = "gpt2"
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))

# Loading the fine-tuned T5 model for question generation
t5_model_name = "valhalla/t5-small-qg-hl"
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Generate response using GPT-2 model
    inputs = gpt2_tokenizer(
        prompt, return_tensors="pt", max_length=100, truncation=True, padding=True
    )

    outputs = gpt2_model.generate(
        inputs.input_ids,
        max_length=250,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.9,
        temperature=0.6,
        repetition_penalty=1.2,
        pad_token_id=gpt2_tokenizer.eos_token_id,
    )

    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Generate question from the AI response using T5 model
    generated_question = generate_question(response)

    return jsonify({"response": response, "question": generated_question})


def generate_question(text):
    input_text = "generate question: " + text
    inputs = t5_tokenizer(input_text, return_tensors="pt")

    try:
        outputs = t5_model.generate(
            inputs["input_ids"], max_length=64, num_beams=4, early_stopping=True
        )

        question = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Generated Question:", question)

        return question

    except Exception as e:
        # Print any errors that occur during question generation
        print("Error during question generation:", str(e))
        return "Error generating question."


# Flask app
if __name__ == "__main__":
    app.run(debug=True)
