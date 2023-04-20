from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import torch

def predict_summary(text):
    #text = "I can't say no It's ripping me apart In a dark room in cold sheets I can't feel a damn thing I lost myself Between your legs Your medicine is in my head You know I'd rather be alone But then you call me on the phone Oh the habits of my heart I can't say no It's ripping me apart You get too close You make it hard to let you go Yeah I tell myself, I like that When you tie my hands behind my back You're confident I'll give you that But if you love yourself, you can fuck yourself Cause I'd rather be alone But you're fragmented in my bones Oh the habits of my heart I can't say no It's ripping me apart You get too close You make it hard to let you go Oh the habits of my heart I can't say no It's ripping me apart You get too close You make it hard to let you go I can't say no It's ripping me apart You get too close"
    text = "summarize: " + text
    checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer(text, return_tensors="pt").input_ids
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    map_location = torch.device('cpu')
    model.load_state_dict(torch.load('predict_summaries/summarizer_model_params.pth', map_location=map_location))
    model = model.cpu()
    outputs = model.generate(inputs, max_length=128, min_length=64, num_beams=8, early_stopping=True, no_repeat_ngram_size=2, length_penalty = 2.0, temperature = 1.5)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary