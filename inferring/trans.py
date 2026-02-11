from transformers import pipeline


def translate(text):
    model_checkpoint = "Helsinki-NLP/opus-mt-ko-en"
    translator = pipeline("translation", model=model_checkpoint)
    translated=translator(text)
    translated=translated[0]["translation_text"]
    return translated