import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from IndicTransToolkit.processor import IndicProcessor
import gradio as gr


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4


def initialize_model_and_tokenizer(ckpt_dir, quantization=None):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(load_in_8bit=True)
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig is None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()
    return tokenizer, model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i: i + BATCH_SIZE]
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated_tokens = model.generate(**inputs, max_length=256, num_beams=5)

        generated_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations


### Initialize everything
ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
tokenizer, model = initialize_model_and_tokenizer(ckpt_dir)
ip = IndicProcessor(inference=True)


### Gradio Interface
def translate_text(text):
    src_lang, tgt_lang = "eng_Latn", "hin_Deva"
    translations = batch_translate([text], src_lang, tgt_lang, model, tokenizer, ip)
    return translations[0]


iface = gr.Interface(fn=translate_text, inputs="text", outputs="text", title="English → Hindi Translation")
iface.launch(server_name="0.0.0.0", server_port=7860)
