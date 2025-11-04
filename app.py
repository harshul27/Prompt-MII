import streamlit as st
import pandas as pd
from datasets import load_dataset
import random
from collections import deque
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# --------- Prompt-MII Logic with Self-Consistency/Votes ---------
def prompt_mii_instruction(subject, choices):
    choices_str = ", ".join(choices)
    return f"Task: {subject}. Choose from [{choices_str}]. Use contextual/logical clues."

def classify_prompt_mii(question, choices, n_votes=5):
    votes = []
    for _ in range(n_votes):
        found = False
        for c in choices:
            for word in question.split():
                if word.lower() in c.lower():
                    votes.append(choices.index(c))
                    found = True
                    break
            if found:
                break
        if not found:
            votes.append(random.randint(0, len(choices)-1))
    return max(set(votes), key=votes.count) if votes else random.randint(0, len(choices)-1)

def random_baseline(choices):
    return random.randint(0, len(choices)-1)

def classic_fewshot_answer(question, choices):
    return random.randint(0, len(choices)-1)

# --------- Exposed App Settings ---------
st.title("Prompt-MII Utility: Train, Benchmark & Demo with Real Datasets")

st.markdown("""
- Test Prompt-MII on built-in, custom, or **live benchmark datasets** like MMLU.
- Train/test with data from Hugging Face Datasets in real time!
- Use the controls below to load a benchmark, run batch testing, and view live leaderboard.
""")

# Loaders and controls
n_votes = st.sidebar.slider("Prompt-MII Self-Consistency/Votes", min_value=1, max_value=21, value=5, step=2)
dataset_source = st.selectbox("Select a Data Source:", ["Manual Entry", "Upload CSV", "MMLU (from Huggingface)"])
mmlu_subset = st.selectbox("MMLU Subset (if chosen):", ["all", "abstract_algebra", "college_biology", "high_school_chemistry", "us_foreign_policy", "international_law"])

# Case buffer
if "case_bank" not in st.session_state:
    st.session_state.case_bank = []

# --------- Dataset Loading Section ---------
batch_df = None
if dataset_source == "Manual Entry":
    st.info("Fill the fields below for single example evaluation & training.")
    subject = st.text_input("Subject", "")
    question = st.text_area("Question", "")
    choices = [c.strip() for c in st.text_input("Choices (comma separated)", "").split(",") if c.strip()]
    correct_answer = st.text_input("Correct Answer (must match a choice)", "")
    if st.button("Evaluate Example"):
        if not subject or not question or not choices or not correct_answer:
            st.warning("Please fill all fields and provide a correct answer.")
        else:
            correct_idx = choices.index(correct_answer)
            mii_pred = classify_prompt_mii(question, choices, n_votes=n_votes)
            classic_pred = classic_fewshot_answer(question, choices)
            rand_pred = random_baseline(choices)
            st.write(f"Prompt-MII Prediction: {choices[mii_pred]} | Acc: {int(mii_pred==correct_idx)} | F1: {f1_score([correct_idx],[mii_pred]):.2f}")
            st.write(f"Classic Prediction: {choices[classic_pred]} | Acc: {int(classic_pred==correct_idx)} | F1: {f1_score([correct_idx],[classic_pred]):.2f}")
            st.write(f"Random Prediction: {choices[rand_pred]} | Acc: {int(rand_pred==correct_idx)} | F1: {f1_score([correct_idx],[rand_pred]):.2f}")
            # Store for utility retrieval
            st.session_state.case_bank.append({
                "subject": subject, "question": question, "choices": choices, "answer": correct_answer
            })
elif dataset_source == "Upload CSV":
    uploaded_csv = st.file_uploader("Upload your dataset (CSV: subject,question,choices,answer)", type=["csv"])
    if uploaded_csv is not None:
        batch_df = pd.read_csv(uploaded_csv)
        st.write(f"Loaded {len(batch_df)} rows.")
elif dataset_source == "MMLU (from Huggingface)":
    num_rows = st.number_input("Rows to sample from MMLU", min_value=10, max_value=200, value=30)
    if "mmlu_df" not in st.session_state or st.button("Load/Reload MMLU sample"):
        st.info("Fetching real examples from Hugging Face…")
        mmlu = load_dataset("cais/mmlu", mmlu_subset, split="test").shuffle(seed=42).select(range(num_rows))
        mmlu_df = pd.DataFrame(mmlu)
        mmlu_df["choices"] = mmlu_df["choices"].apply(lambda x: [str(y) for y in x])
        mmlu_df["answer_str"] = mmlu_df.apply(lambda row: row["choices"][int(row["answer"])] if str(row["answer"]).isdigit() else row["answer"], axis=1)
        st.session_state.mmlu_df = mmlu_df
    batch_df = st.session_state.get("mmlu_df", None)
    if batch_df is not None:
        st.write(batch_df.head())

# --------- Batch Inference ---------
if batch_df is not None:
    st.markdown("Click 'Run Batch Evaluation' to compute end-to-end validation on the chosen dataset.")
    if st.button("Run Batch Evaluation"):
        results = []
        for idx, row in batch_df.iterrows():
            subject = row["subject"] if "subject" in row else ""
            question = row["question"]
            choices = row["choices"]
            answer = row["answer_str"] if "answer_str" in row else row.get("answer", "")
            if type(choices) is str:
                try:
                    choices = eval(choices) # for CSV stringified lists
                except:
                    choices = [c.strip() for c in choices.split(",")]
            if answer not in choices:
                continue
            correct_idx = choices.index(answer)
            mii_pred = classify_prompt_mii(question, choices, n_votes=n_votes)
            classic_pred = classic_fewshot_answer(question, choices)
            rand_pred = random_baseline(choices)
            y_true = [correct_idx]
            mii_f1 = f1_score(y_true, [mii_pred], average='macro')
            classic_f1 = f1_score(y_true, [classic_pred], average='macro')
            rand_f1 = f1_score(y_true, [rand_pred], average='macro')
            results.append({
                "idx": idx,
                "q": question,
                "ans": answer,
                "mii_acc": int(mii_pred == correct_idx),
                "classic_acc": int(classic_pred == correct_idx),
                "rand_acc": int(rand_pred == correct_idx),
                "mii_f1": mii_f1,
                "classic_f1": classic_f1,
                "rand_f1": rand_f1
            })
        r = pd.DataFrame(results)
        st.dataframe(r.head(10))
        st.success(f"Results on {len(r)} samples for {dataset_source}!")

        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(12,5))
        for label, key in [("Prompt-MII", "mii_acc"), ("Classic", "classic_acc"), ("Random", "rand_acc")]:
            axs[0].plot(r[key].cumsum() / (r.index + 1), label=label)
        axs[0].set_title("Cumulative Accuracy")
        axs[0].set_ylabel("Accuracy")
        axs[0].set_ylim(0,1.05)
        axs[0].legend()
        for label, key in [("Prompt-MII", "mii_f1"), ("Classic", "classic_f1"), ("Random", "rand_f1")]:
            axs[1].plot(r[key], label=label)
        axs[1].set_title("F1 Score per Example")
        axs[1].set_ylabel("F1")
        axs[1].set_ylim(0,1.05)
        axs[1].legend()
        st.pyplot(fig)
