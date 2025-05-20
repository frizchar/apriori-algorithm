from flask import Flask, request, render_template_string
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

# Sample grocery transactions dataset (one-hot encoded)
data = {
    'Eggs':    [1, 0, 1, 1, 0, 1, 0, 1],
    'Milk':    [1, 1, 1, 0, 1, 1, 0, 0],
    'Bread':   [0, 1, 1, 1, 0, 0, 1, 1],
    'Butter':  [1, 0, 1, 0, 1, 0, 1, 0],
    'Cheese':  [0, 1, 0, 1, 1, 0, 1, 1],
    'Diaper':  [1, 0, 0, 1, 0, 1, 1, 0],
    'Beer':    [0, 1, 0, 0, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

html = '''
<!doctype html>
<html>
<head>
  <title>Apriori on Grocery Transactions</title>
  <style>
    body {
      background-color: #dbe5be;
      color: #000080;
      font-family: Arial, sans-serif;
      padding: 20px;
    }
    textarea {
      width: 100%;
      height: 300px;
      font-family: monospace;
      white-space: pre-wrap;
      background: #f9f9f9;
      border: 1px solid #ccc;
      padding: 10px;
      margin-top: 10px;
    }
    input[type=submit] {
      background-color: #000080;
      color: white;
      border: none;
      padding: 10px 20px;
      cursor: pointer;
      font-size: 1em;
    }
    input[type=submit]:hover {
      background-color: #003366;
    }
  </style>
</head>
<body>
  <h1>Apriori Algorithm on Sample Grocery Transactions</h1>
  <form method="POST">
    Minimum Support (0-1): <input type="number" name="min_support" step="0.01" min="0" max="1" value="0.3" required>
    <input type="submit" value="Run Apriori">
  </form>
  {% if frequent_itemsets %}
    <h2>Frequent Itemsets (Support ≥ {{ min_support }})</h2>
    <textarea readonly>{{ frequent_itemsets }}</textarea>
  {% endif %}
  {% if rules %}
    <h2>Association Rules (Confidence ≥ 0.7)</h2>
    <textarea readonly>{{ rules }}</textarea>
  {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def apriori_groceries():
    frequent_itemsets_str = None
    rules_str = None
    min_support = 0.3  # default

    if request.method == 'POST':
        try:
            min_support = float(request.form['min_support'])
            if not (0 < min_support <= 1):
                raise ValueError("Support must be between 0 and 1")

            # Find frequent itemsets
            frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

            # Generate association rules with confidence threshold 0.7
            rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

            # Format frequent itemsets for display
            frequent_itemsets_str = frequent_itemsets.to_string(index=False)

            # Format rules for display
            if not rules_df.empty:
                rules_list = []
                for _, row in rules_df.iterrows():
                    antecedents = ', '.join(row['antecedents'])
                    consequents = ', '.join(row['consequents'])
                    conf = row['confidence']
                    supp = row['support']
                    lift = row['lift']
                    rules_list.append(f"{antecedents} -> {consequents} (conf: {conf:.2f}, supp: {supp:.2f}, lift: {lift:.2f})")
                rules_str = "\n".join(rules_list)
            else:
                rules_str = "No association rules found with the given confidence threshold."

        except Exception as e:
            frequent_itemsets_str = f"Error: {e}"

    return render_template_string(html, frequent_itemsets=frequent_itemsets_str, rules=rules_str, min_support=min_support)


if __name__ == '__main__':
    app.run(debug=True)
