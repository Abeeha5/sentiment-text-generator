import streamlit as st
from generate_text import generate_text
from sentiment import sentiment_analyzer

# initialize
st.set_page_config(page_title="AI Sentiment Text Generator", layout="centered")
st.title("AI Text Generator — Sentiment-aware")
st.markdown("Enter a prompt. The model will detect the sentiment and generate a paragraph aligned with it.")

@st.cache_resource
def get_analyzer():
    return sentiment_analyzer()

@st.cache_resource
def get_generator():
    return generate_text()

analyzer = get_analyzer()
generator = get_generator()

#  UI 
prompt = st.text_area("Prompt", value="Write a prompt.", height=120)

generate_button = st.button("Generate")

if generate_button:
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        # Sentiment analysis
        label, score= analyzer.assign_labels(prompt)
        detected = label
        
        st.info(f"Detected sentiment: **{detected}** (classifier label: {label}, confidence: {score:.2f})")

        with st.spinner("Generating..."):
            try:
                output = generator.generate(prompt, sentiment=detected)
                st.subheader("Generated text")
                st.write(output)
            except Exception as e:
                st.error(f"Generation failed: {e}")

# Footer
st.markdown("---")
st.markdown("""
**Notes**
- Uses a pre-trained DistilBERT sentiment classifier and GPT-2 for text generation.
""")