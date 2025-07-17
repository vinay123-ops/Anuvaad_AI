import gradio as gr
from ai4bharat.transliteration import XlitEngine

# Initialize the engine for Tamil (ta) and Malayalam (ml) as examples
engine = XlitEngine(["ta", "ml"], beam_width=6, src_script_type="en")


def transliterate_text(text, lang):
    """
    Transliterate the input text into the selected language.
    """
    engine_lang = XlitEngine(lang, beam_width=6, src_script_type="en")
    output = engine_lang.translit_sentence(text)
    return output


with gr.Blocks() as demo:
    gr.Markdown("## 📝 English to Indic Transliteration (Powered by AI4Bharat)")
    
    with gr.Row():
        input_text = gr.Textbox(label="Input (English)")
        language = gr.Dropdown(
            choices=[
                "hi", "ta", "te", "ml", "mr", "bn", "gu", "pa", "or", "as", "kn", 
                "sa", "sd", "ks", "ne", "si", "ur", "brx", "gom", "mai", "mni"
            ],
            label="Target Language (ISO 639-1 / AI4Bharat Codes)",
            value="ta"
        )
    output_text = gr.Textbox(label="Transliterated Output")

    btn = gr.Button("Transliterate")
    btn.click(fn=transliterate_text, inputs=[input_text, language], outputs=output_text)

demo.launch(share=True)
