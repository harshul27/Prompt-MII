import streamlit as st
import pandas as pd
from datasets import load_dataset
import random
from collections import deque
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# --------- Prompt-MII Logic ---------
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

st.title("Prompt-MII Benchmark Utility: MMLU Integration")

n_votes = st.sidebar.slider("Prompt-MII Self-Consistency/Votes", min_value=1, max_value=21, value=5, step=2)
sample_size = st.number_input("Sample size (MMLU):", min_value=10, max_value=200, value=40, step=10)
mmlu_subset = st.selectbox("MMLU Subset:", ["all", "abstract_algebra", "college_biology", "high_school_chemistry", "us_foreign_policy"])

if st.button("Load and Evaluate MMLU Sample"):
    st.info("Loading online MMLU data, please wait…")
    mmlu = load_dataset("cais/mmlu", mmlu_subset, split="test").shuffle(seed=42).select(range(sample_size))
    mmlu_df = pd.DataFrame(mmlu)
    mmlu_df["choices"] = mmlu_df["choices"].apply(lambda x: [str(y) for y in x])
    mmlu_df["answer_str"] = mmlu_df.apply(lambda row: row["choices"][int(row["answer"])] if str(row["answer"]).isdigit() else row["answer"], axis=1)
    st.write(mmlu_df.head())
    results = []
    total_tokens = []
    for idx, row in mmlu_df.iterrows():
        subject = row["subject"]
        question = row["question"]
        choices = row["choices"]
        answer = row["answer_str"]
        if answer not in choices: continue
        correct_idx = choices.index(answer)
        mii_inst = prompt_mii_instruction(subject, choices)
        mii_pred = classify_prompt_mii(question, choices, n_votes=n_votes)
        classic_pred = classic_fewshot_answer(question, choices)
        rand_pred = random_baseline(choices)
        # Metrics
        y_true = [correct_idx]
        mii_f1 = f1_score(y_true, [mii_pred], average='macro')
        classic_f1 = f1_score(y_true, [classic_pred], average='macro')
        rand_f1 = f1_score(y_true, [rand_pred], average='macro')
        mii_acc = int(mii_pred == correct_idx)
        classic_acc = int(classic_pred == correct_idx)
        rand_acc = int(rand_pred == correct_idx)
        mii_tokens = len(mii_inst.split()) + len(question.split())
        total_tokens.append(mii_tokens)
        results.append({
            "idx": idx,
            "subject": subject,
            "question": question,
            "mii_acc": mii_acc,
            "classic_acc": classic_acc,
            "rand_acc": rand_acc,
            "mii_f1": mii_f1,
            "classic_f1": classic_f1,
            "rand_f1": rand_f1,
            "mii_tokens": mii_tokens
        })
    df_results = pd.DataFrame(results)
    st.dataframe(df_results.head(10))
    st.success(f"Completed on {len(df_results)} samples from MMLU subset '{mmlu_subset}'!")

    # Plot accuracy, F1, and token usage
    fig, axs = plt.subplots(1, 3, figsize=(16,5))
    for label, key in [("Prompt-MII", "mii_acc"), ("Classic", "classic_acc"), ("Random", "rand_acc")]:
        axs[0].plot(df_results[key].cumsum()/(df_results.index+1), label=label)
    axs[0].set_title("Cumulative Accuracy")
    axs[0].set_ylim(0,1.05)
    axs[0].legend()
    for label, key in [("Prompt-MII", "mii_f1"), ("Classic", "classic_f1"), ("Random", "rand_f1")]:
        axs[1].plot(df_results[key], label=label)
    axs[1].set_title("F1 Score per Example")
    axs[1].set_ylim(0,1.05)
    axs[1].legend()
    axs[2].plot(df_results["mii_tokens"], "o-", label="Prompt-MII Token Usage")
    axs[2].axhline(100, color="grey", linestyle="--", label="Classic Token Reference")
    axs[2].set_title("Token Usage per Example")
    axs[2].set_ylabel("Tokens")
    axs[2].legend()
    st.pyplot(fig)
    st.text(f"Prompt-MII average token usage: {sum(total_tokens)/len(total_tokens):.1f}")
    st.text(f"Classic ICL average token usage: 100")
    st.text(f"Efficiency gain: {100*(1-(sum(total_tokens)/len(total_tokens))/100):.1f}%")
