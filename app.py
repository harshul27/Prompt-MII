import streamlit as st
import random
from sklearn.metrics import f1_score, accuracy_score
import openai

# --------- PROMPT-MII LOGIC ---------
def prompt_mii_instruction(subject, choices):
    choices_str = ", ".join(choices)
    return f"Task: {subject}. Choose from [{choices_str}]. Use contextual/logical clues."

def classify_prompt_mii(question, choices):
    # Simulated Prompt-MII
    for c in choices:
        for word in question.split():
            if word.lower() in c.lower():
                return choices.index(c)
    return random.randint(0, len(choices)-1)

def random_baseline(choices):
    return random.randint(0, len(choices)-1)

def classic_fewshot_answer(question, choices):
    # Simulated classic few-shot
    return random.randint(0, len(choices)-1)

def openai_prompt_inference(prompt, choices, api_key, model="gpt-3.5-turbo"):
    openai.api_key = api_key
    messages = [
        {"role": "system", "content": "You are a helpful, accurate multiple-choice assistant."},
        {"role": "user", "content": f"{prompt}\nChoices: {', '.join(choices)}."}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=20,
        temperature=0
    )
    output = response.choices[0].message['content'].strip()
    # Match best choice (substring match fallback)
    best_choice = max(choices, key=lambda c: output.lower().count(c.lower()))
    idx = choices.index(best_choice)
    return idx, output

# --------- STREAMLIT INTERFACE ---------
st.title("Prompt-MII vs Classic Prompting: Real-Time AI Prompt Demo")
st.markdown("""
Test Prompt-MII, Classic (few-shot), and Random Baseline —  
**Try live with a real OpenAI LLM by entering your API key!**  
Measure predictions, token efficiency, and see synthesized vs classic instructions side-by-side.
""", unsafe_allow_html=True)

subject = st.selectbox("Select a Subject", [
    "Finance", "Science", "Customer Support", "History", "Law", "E-Commerce", "Other"
])
question = st.text_area("Enter your Question/Task")
choices = [c.strip() for c in st.text_input("Comma-separated Answer Choices", "Yes,No,Not Sure,Irrelevant").split(",") if c.strip()]
correct_answer = st.text_input("Correct Answer (must match one choice exactly)")

api_key = st.text_input("OPTIONAL: Enter your OpenAI API Key (for live LLM):", type="password")
use_live_llm = st.checkbox("Use OpenAI LLM for Prompt-MII (requires API key)")

if st.button("Run Demo"):
    if not question or not choices or not correct_answer:
        st.warning("Please enter a question, choices, and the correct answer.")
    else:
        # Find true index for accuracy calculation
        try:
            correct_idx = choices.index(correct_answer)
        except ValueError:
            st.error("Correct answer must exactly match one of the choices!")
            st.stop()

        col1, col2, col3, col4 = st.columns(4)
        # Run predictors
        mii_pred = classify_prompt_mii(question, choices)
        random_pred = random_baseline(choices)
        classic_pred = classic_fewshot_answer(question, choices)
        mii_instruction = prompt_mii_instruction(subject, choices)
        classic_instruction = "Please read the question carefully and choose the best answer."

        col1.metric("Prompt-MII (Sim)", choices[mii_pred])
        col2.metric("Classic (Sim)", choices[classic_pred])
        col3.metric("Random", choices[random_pred])

        # Accuracy display (1 if matches correct, 0 if not)
        col1.metric("Prompt-MII Accuracy", f"{int(mii_pred == correct_idx)}")
        col2.metric("Classic Accuracy", f"{int(classic_pred == correct_idx)}")
        col3.metric("Random Accuracy", f"{int(random_pred == correct_idx)}")

        # Live LLM with OpenAI
        if use_live_llm and api_key:
            try:
                with st.spinner("Calling OpenAI LLM (Prompt-MII instruction)..."):
                    openai_idx, openai_raw = openai_prompt_inference(
                        f"{mii_instruction}\nQuestion: {question}",
                        choices, api_key
                    )
                col4.metric("Prompt-MII (OpenAI LLM)", choices[openai_idx])
                col4.metric("OpenAI Accuracy", f"{int(openai_idx == correct_idx)}")
                st.code(f"OpenAI LLM raw: {openai_raw}", language="markdown")
            except Exception as e:
                st.error(f"OpenAI API error: {e}")

        st.subheader("Prompt-MII Synthesized Instruction")
        st.code(mii_instruction)

        st.subheader("Classic ICL Prompt Format")
        st.code(classic_instruction)

        # Token bar chart
        st.markdown("### Token Efficiency (Cost Impact)")
        mii_tokens = len(mii_instruction.split()) + len(question.split())
        classic_tokens = 100   # You can adjust for your classic prompt size
        random_tokens = 1
        bar_data = {"Prompt-MII": [mii_tokens], "Classic ICL": [classic_tokens], "Random": [random_tokens]}
        if use_live_llm and api_key:
            bar_data["OpenAI (Prompt-MII)"] = [mii_tokens]
        st.bar_chart(bar_data)
        st.caption(f"Prompt-MII uses {mii_tokens} tokens, Classic ICL {classic_tokens} tokens ({100*(1-mii_tokens/classic_tokens):.1f}% saved).")

st.caption("Built for live, business-facing AI demos. Now with per-query accuracy! Add OpenAI key for real LLM answers.")
