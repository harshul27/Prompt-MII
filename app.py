import streamlit as st
import random
from sklearn.metrics import f1_score, accuracy_score

# --------- PROMPT-MII LOGIC ---------
def prompt_mii_instruction(subject, choices):
    choices_str = ", ".join(choices)
    return f"Task: {subject}. Choose from [{choices_str}]. Use contextual/logical clues."

def classify_prompt_mii(question, choices):
    for c in choices:
        for word in question.split():
            if word.lower() in c.lower():
                return choices.index(c)
    return random.randint(0, len(choices)-1)

def random_baseline(choices):
    return random.randint(0, len(choices)-1)

def classic_fewshot_answer(question, choices):
    # Stub: simulate classic few-shot with random pick (replace with API call if desired)
    return random.randint(0, len(choices)-1)

# --------- STREAMLIT INTERFACE ---------
st.title("Prompt-MII vs Classic Prompting: Real-Time AI Prompt Demo")
st.markdown("""
Enter a question, choose a subject/domain, and see how Prompt-MII performs compared to classic few-shot (“ICL”) and a random baseline.
""")

subject = st.selectbox("Select a Subject", [
    "Finance", "Science", "Customer Support", "History", "Law", "E-Commerce", "Other"
])
question = st.text_area("Enter your Question/Task")
choices = st.text_input("Comma-separated Answer Choices", "Yes,No,Not Sure,Irrelevant").split(",")

if st.button("Run Demo"):
    if not question:
        st.warning("Please enter a question.")
    else:
        # Make predictions
        mii_pred = classify_prompt_mii(question, choices)
        random_pred = random_baseline(choices)
        classic_pred = classic_fewshot_answer(question, choices)
        
        # Synthesize instructions
        mii_instruction = prompt_mii_instruction(subject, choices)
        classic_instruction = "Please read the question carefully and choose the best answer."

        # Display results
        col1, col2, col3 = st.columns(3)
        col1.metric("Prompt-MII Prediction", choices[mii_pred])
        col2.metric("Classic ICL Prediction", choices[classic_pred])
        col3.metric("Random Baseline", choices[random_pred])
        
        st.subheader("Prompt-MII Synthesized Instruction")
        st.code(mii_instruction)
        
        st.subheader("Classic ICL Prompt Format")
        st.code(classic_instruction)
        
        st.markdown("Token Efficiency (cost impact):")
        mii_tokens = len(mii_instruction.split()) + len(question.split())
        classic_tokens = 100  # Simulate 100-token context
        random_tokens = 1
        st.bar_chart({"Prompt-MII": [mii_tokens], "Classic ICL": [classic_tokens], "Random": [random_tokens]})
        st.markdown(f"**Prompt-MII uses {mii_tokens} tokens, Classic ICL {classic_tokens} tokens ({100*(1-mii_tokens/classic_tokens):.1f}% saved)**")
        
        st.info("This demo simulates Prompt-MII's logic. For a full business use case, connect to your company LLMs and calculations.")

st.caption("Built for real-world business demos. Adapt and extend for your team or project!")
