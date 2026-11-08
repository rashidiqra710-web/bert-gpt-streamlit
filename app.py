import html
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="BERT vs GPT-2", page_icon="🤖", layout="centered")
st.title("Encoder-Only vs Decoder-Only Transformers")

section = st.sidebar.radio(
    "Choose a section",
    ["1. BERT (Encoder-only): Multilingual NER",
     "2. GPT-2 (Decoder-only): Balochi Generation"]
)

# ---------- Model loaders ----------
@st.cache_resource
def load_bert():
    return pipeline("token-classification",
                    model="Davlan/bert-base-multilingual-cased-ner-hrl",
                    aggregation_strategy="simple")

@st.cache_resource
def load_gpt():
    return pipeline("text-generation", model="rashidiqra/gpt2-balochi")

# ---------- Section 1: BERT ----------
if section.startswith("1"):
    st.header("BERT: Named Entity Recognition")
    st.write("Encoder-only model. It reads the whole sentence and labels "
             "people (PER), organizations (ORG), locations (LOC) and dates (DATE). "
             "Works in many high-resource languages.")

    examples = {
        "English": "Elon Musk visited Google in California last Monday.",
        "German": "Angela Merkel besuchte Berlin im Jahr 2015.",
        "Spanish": "Lionel Messi juega en Miami con el equipo Inter Miami.",
        "French": "Emmanuel Macron a rencontré des ministres à Paris.",
        "Russian": "Владимир Путин приехал в Москву.",
        "Chinese": "马云在杭州创立了阿里巴巴。",
        "Arabic": "زار محمد صلاح مدينة ليفربول.",
    }
    choice = st.selectbox("Pick an example language", list(examples.keys()))
    text = st.text_area("Input sentence (you can edit it)", examples[choice])

    if st.button("Run BERT"):
        with st.spinner("Loading BERT and analyzing..."):
            ner = load_bert()
            results = ner(text)
        if not results:
            st.info("No entities found.")
        else:
            st.table([{"Entity": r["word"],
                       "Type": r["entity_group"],
                       "Confidence": round(float(r["score"]), 2)} for r in results])

# ---------- Section 2: GPT-2 ----------
else:
    st.header("GPT-2: Balochi Text Generation")
    st.write("Decoder-only model fine-tuned on Balochi (low-resource, monolingual). "
             "It uses the previous words to predict and generate the next words.")

    prompt = st.text_area("Start a Balochi sentence", "من اقرہ آں شہ")
    length = st.slider("Number of new tokens", 10, 100, 40)
    temp = st.slider("Creativity (temperature)", 0.3, 1.2, 0.8)

    if st.button("Generate"):
        if not prompt.strip():
            st.warning("Please write the beginning of a Balochi sentence.")
        else:
            with st.spinner("Loading GPT-2 and generating..."):
                gen = load_gpt()
                out = gen(prompt, max_new_tokens=length, do_sample=True,
                          top_k=50, top_p=0.95, temperature=temp,
                          no_repeat_ngram_size=3)
            st.markdown(
                f"<div dir='rtl' style='font-size:22px'>"
                f"{html.escape(out[0]['generated_text'])}</div>",
                unsafe_allow_html=True)