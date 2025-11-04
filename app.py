import streamlit as st
import pandas as pd
from datasets import load_dataset
import random
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# === Prompt Construction ===
def prompt_mii_instruction(subject, choices):
    # Meta-learned, token-efficient
    return f"Task: {subject}. Choose from [{', '.join(choices)}]. Use contextual/logical clues."

def classic_fewshot_prompt(subject, question, choices, shots=4):
    # Simulate real few-shot: multiple examples
    fewshot_examples = [
        f"Q{i+1}: Example about {subject}? Choices: {', '.join(choices)}. A: {random.choice(choices)}"
        for i in range(shots)
    ]
    fewshot_block = "\n".join(fewshot_examples)
    return f"{fewshot_block}\nQ: {question}\nChoices: {', '.join(choices)}"

def zero_shot_prompt(subject, question, choices):
    # Industry zero-shot: plain question, choices
    return f"Subject: {subject}\nQ: {question}\nChoices: {', '.join(choices)}"

def chain_of_thought_prompt(subject, question, choices):
    # Widely used: add “think step by step”
    return f"Subject: {subject}\nQ: {question}\nChoices: {', '.join(choices)}\nLet's think step by step."

def random_baseline(choices):
    return random.randint(0, len(choices)-1)

def majority_baseline(choices, batch_answers):
    maj_ans = max(set(batch_answers), key=batch_answers.count)
    return choices.index(maj_ans) if maj_ans in choices else random_baseline(choices)

def count_tokens(text):
    # Simple proxy: word count.
    return len(text.split())

# ===========================
st.set_page_config(layout="wide")
st.title("🧪 Prompt Engineering Benchmark: MII, Few-shot, Zero-shot, Chain-of-Thought, and Baselines")
st.markdown("""
- Explore true token-efficient meta-prompting (Prompt-MII) vs leading industry strategies.
- See accuracy, F1, and token efficiency in action.
---
""")

n_votes = st.sidebar.slider("Prompt-MII Voting Ensemble", min_value=1, max_value=15, value=5, step=2)
sample_size = st.sidebar.number_input("Number of Samples", min_value=20, max_value=1000, value=100, step=10)
mmlu_subset = st.sidebar.selectbox("MMLU Domain", ["all", "business_ethics", "college_biology", "us_foreign_policy"])
fewshot_shots = st.sidebar.slider("Classic Few-Shot - #examples", min_value=2, max_value=8, value=4, step=1)

if st.button("Run Prompt Engineering Benchmark"):
    mmlu = load_dataset("cais/mmlu", mmlu_subset, split="test")
    mmlu = mmlu.shuffle(seed=123).select(range(sample_size))
    mmlu_df = pd.DataFrame(mmlu)
    mmlu_df["choices"] = mmlu_df["choices"].apply(lambda x: [str(y) for y in x])
    mmlu_df["answer_str"] = mmlu_df.apply(
        lambda row: 
            row["choices"][int(row["answer"])] if (str(row["answer"]).isdigit() and int(row["answer"]) < len(row["choices"])) 
            else row.get("answer", ""), axis=1)

    # Error reporting
    invalid_idxs = []
    batch_answers = []
    for idx, row in mmlu_df.iterrows():
        ans = row["answer_str"]
        ch = row["choices"]
        if ans not in ch:
            invalid_idxs.append(idx)
        else:
            batch_answers.append(ans)
    mmlu_df_valid = mmlu_df.drop(index=invalid_idxs)
    st.info(f"Excluded {len(invalid_idxs)} invalid samples.")

    results = []
    y_true = []
    y_mii, y_classic, y_zero, y_cot, y_maj, y_rand = [], [], [], [], [], []

    for idx, row in mmlu_df_valid.iterrows():
        subject = row["subject"]
        question = row["question"]
        choices = row["choices"]
        answer = row["answer_str"]
        correct_idx = choices.index(answer)
        y_true.append(correct_idx)

        # PROMPT CONSTRUCTION & FAKE INFERENCE (replace w/ LLM API for real!)
        mii_prompt = prompt_mii_instruction(subject, choices) + "\nQ: " + question + "\nChoices: " + ", ".join(choices)
        mii_pred = random_baseline(choices)  # Replace with actual Prompt-MII LLM inference
        classic_prompt = classic_fewshot_prompt(subject, question, choices, shots=fewshot_shots)
        classic_pred = random_baseline(choices)  # Replace with actual LLM
        zero_prompt = zero_shot_prompt(subject, question, choices)
        zero_pred = random_baseline(choices)    # Replace with actual LLM
        cot_prompt = chain_of_thought_prompt(subject, question, choices)
        cot_pred = random_baseline(choices)     # Replace with actual LLM
        maj_pred = majority_baseline(choices, batch_answers)
        rand_pred = random_baseline(choices)

        y_mii.append(mii_pred)
        y_classic.append(classic_pred)
        y_zero.append(zero_pred)
        y_cot.append(cot_pred)
        y_maj.append(maj_pred)
        y_rand.append(rand_pred)

        results.append({
            "idx": idx, "subject": subject, "question": question, "choices": "|".join(choices), "answer": answer,
            "Prompt-MII": choices[mii_pred], "Classic-FewShot": choices[classic_pred], "Zero-Shot": choices[zero_pred],
            "Chain-of-Thought": choices[cot_pred], "Majority": choices[maj_pred], "Random": choices[rand_pred],
            "mii_tokens": count_tokens(mii_prompt),
            "classic_tokens": count_tokens(classic_prompt),
            "zero_tokens": count_tokens(zero_prompt),
            "cot_tokens": count_tokens(cot_prompt)
        })

    # Macro metrics & summary
    methods = {
        "Prompt-MII": (y_mii, "mii_tokens"),
        "Classic-FewShot": (y_classic, "classic_tokens"),
        "Zero-Shot": (y_zero, "zero_tokens"),
        "Chain-of-Thought": (y_cot, "cot_tokens"),
        "Majority": (y_maj, None),
        "Random": (y_rand, None)
    }
    macro_acc = {name: accuracy_score(y_true, preds) for name, (preds, _) in methods.items()}
    macro_f1  = {name: f1_score(y_true, preds, average='macro') for name, (preds, _) in methods.items()}
    avg_tokens = {name: (sum([r[tok_col] for r in results]) / len(results) if tok_col and len(results) > 0 else None) for name, (_, tok_col) in methods.items()}
    
    # Boardroom summary
    st.markdown("### Benchmark Metrics (Macro-Accuracy, F1, Token Usage)")
    rows = []
    for name in methods.keys():
        t = avg_tokens[name]
        t_text = f"{t:.1f}" if t is not None else "N/A"
        rows.append([name, f"{macro_acc[name]:.2%}", f"{macro_f1[name]:.2%}", t_text])
    st.dataframe(pd.DataFrame(rows, columns=["Technique", "Accuracy", "F1", "Avg. Tokens"]), use_container_width=True)

    # Plots
    fig, axs = plt.subplots(1, 3, figsize=(18,6))
    for name, (preds, _) in methods.items():
        axs[0].plot(pd.Series(preds).expanding().apply(lambda x: accuracy_score(y_true[:len(x)], x)), label=name)
        axs[1].plot(pd.Series(preds).expanding().apply(lambda x: f1_score(y_true[:len(x)], x, average='macro')), label=name)
    axs[0].set_title("Cumulative Accuracy")
    axs[0].set_ylim(0,1.01)
    axs[0].legend()
    axs[1].set_title("Cumulative Macro F1")
    axs[1].set_ylim(0,1.01)
    axs[1].legend()
    for name, (_, tok_col) in methods.items():
        if tok_col:
            axs[2].plot([r[tok_col] for r in results], 'o', label=name)
    axs[2].set_title("Token Usage per Query")
    axs[2].set_ylabel("Tokens")
    axs[2].legend()
    st.pyplot(fig)

    # First 5 Sample Outputs
    st.markdown("#### Results: Sample Outputs (First 5)")
    st.dataframe(pd.DataFrame(results[:5]), use_container_width=True)

    with st.expander("Prompt Engineering Technique Details and Reasoning"):
        st.write("""
- **Prompt-MII:** Ultra-compressed, meta-learned instruction (token-efficient, robust, high generalization).
- **Classic Few-Shot:** True few-shot, showing real prompt size/cost with multi-example in-context learning.
- **Zero-Shot:** Popular, cost-savvy, task-only prompting.
- **Chain-of-Thought:** Reasoning-driven, widely used for difficult tasks (adds 'think step by step').
- **Majority:** Industry baseline (most frequent label).
- **Random:** Sanity/lower-bound.
- Customize the sample size, ensemble votes, and few-shot K to fit your real-world use case or cost profile.
*Replace random prediction logic with real LLM model calls for each regime for production research or deployment.*
""")
