import json
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

##############################################################################
# Step A: Read all voting_records.jsonl files into a single DataFrame
##############################################################################

def read_voting_records(root_dir="voting_records"):
    """
    Recursively read all .jsonl files under `root_dir`,
    each file containing a single JSON array of records.

    Returns a list of dicts, each with:
      {
        "response_A": str,
        "response_B": str,
        "Won": str,
        "question_id": int,
        "data_id": str
      }
    """
    all_records = []
    # e.g. root_dir might be "voting_records"
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jsonl"):
                full_path = os.path.join(subdir, file)
                with open(full_path, "r", encoding="utf-8") as f:
                    # The file is lines containing JSON arrays.
                    # Usually it’s stored as a single line:
                    # [
                    #   { "response_A": "gpt4o", "response_B": "claude", "Won": "claude", ...},
                    #   ...
                    # ]
                    content = f.read().strip()
                    # Make sure it’s parseable as JSON
                    try:
                        data = json.loads(content)
                        # data should be a single Python list
                        if isinstance(data, list):
                            all_records.extend(data)
                    except json.JSONDecodeError:
                        print(f"Skipping malformed file: {full_path}")
    return all_records


##############################################################################
# Step B: Convert records into a DataFrame suitable for Bradley–Terry
##############################################################################

def build_pairwise_df(records):
    """
    records is a list of dicts of the form:
      {
        "response_A": "modelA",
        "response_B": "modelB",
        "Won": "modelA" or "modelB",
        "question_id": ...,
        "data_id": ...
      }

    Returns a DataFrame with columns: [model_A, model_B, winner].
    Ties are excluded in this simple version.
    """
    rows = []
    for r in records:
        winner = r.get("Won", "")
        if winner == r["response_A"]:
            rows.append({"model_A": r["response_A"],
                         "model_B": r["response_B"],
                         "winner": r["response_A"]})
        elif winner == r["response_B"]:
            rows.append({"model_A": r["response_A"],
                         "model_B": r["response_B"],
                         "winner": r["response_B"]})
        # If it’s a tie or unrecognized, skip it

    return pd.DataFrame(rows, columns=["model_A", "model_B", "winner"])


##############################################################################
# Step C: Build a Bradley–Terry design matrix and fit logistic regression
##############################################################################

def construct_design_matrix(df):
    """
    df has columns: "model_A", "model_B", "winner".
    We gather the unique models, assign them integer indices.
    For each row i:
      X[i, idxA] = +1
      X[i, idxB] = -1
      y[i] = 1 if A is the winner, else 0

    Returns:
      X (n x p)   # design matrix
      y (n,)      # 0/1 outcomes
      model_index # a pd.Series mapping "model_name" -> index
    """
    # Collect all unique model names
    all_models = pd.concat([df["model_A"], df["model_B"]]).unique()
    model_index = pd.Series(data=range(len(all_models)), index=all_models)

    n = len(df)
    p = len(all_models)
    X = np.zeros((n, p))
    y = np.zeros(n)

    for i, row in df.iterrows():
        A = row["model_A"]
        B = row["model_B"]
        winner = row["winner"]

        idxA = model_index[A]
        idxB = model_index[B]

        # A is coded as +1, B as -1
        X[i, idxA] = +1
        X[i, idxB] = -1

        # If A is the winner, y=1, else y=0
        if winner == A:
            y[i] = 1.0
        else:
            y[i] = 0.0

    return X, y, model_index


def fit_bradley_terry(X, y):
    """
    Fit logistic regression without intercept to approximate
    the Bradley–Terry model. 
    Returns the "rating" for each model (lr.coef_[0]).
    """
    lr = LogisticRegression(fit_intercept=False)
    lr.fit(X, y)
    # shape is (1, p)
    return lr.coef_[0]


##############################################################################
# Step D: Putting it all together
##############################################################################

def main(root_dir="ai_and_work/voting_records"):
    # A) Read
    records = read_voting_records(root_dir)
    print(f"Loaded {len(records)} total pairwise comparisons.")

    # B) Build DataFrame
    df = build_pairwise_df(records)
    print(f"After dropping ties/unrecognized: {len(df)} comparisons remain.")

    if df.empty:
        print("No valid pairwise data found. Exiting.")
        return

    # C) Construct design matrix & fit
    X, y, model_index = construct_design_matrix(df)
    ratings = fit_bradley_terry(X, y)

    # D) Sort + show final ranking
    idx_to_model = {v: k for k, v in model_index.to_dict().items()}
    rating_items = [
        (idx_to_model[i], ratings[i]) for i in range(len(ratings))
    ]
    # Sort descending by rating
    rating_items.sort(key=lambda x: x[1], reverse=True)

    print("\n===== Bradley–Terry Ranking =====")
    for rank, (model_name, rating) in enumerate(rating_items, start=1):
        print(f"{rank}. {model_name} (score={rating:.3f})")


if __name__ == "__main__":
    main()
 