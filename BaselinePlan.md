# Baseline Plan: Dataset Plagiarism and Synthetic Data Detection

## 1. Project Objective

The goal of this project is to build a baseline framework to detect whether a submitted dataset is likely to be:

- Collected from real-world respondents
- Fabricated manually
- Generated using an LLM
- Generated using pseudo-random or synthetic data generation methods
- Copied or mass-submitted by multiple teams

This system should not claim to conclusively prove plagiarism or fabrication. Instead, it should work as a **red-flag detection and workflow-quality evaluation system**.

The core idea is simple: machine learning and statistical analysis are fundamentally about pattern detection. If a dataset is artificially generated, copied, overly cleaned, or mechanically produced, it often leaves detectable traces in structure, distribution, timing, repetition, entropy, and workflow behavior.

---

## 2. Important Principle

No automated method can definitively prove that a dataset is LLM-generated or fabricated.

The system should be used as:

- A red-flag generator
- A sanity-checking tool
- A supporting evidence system
- A workflow-quality evaluator
- A decision-support tool for judges or organizers

Final judgment should combine:

- Statistical evidence
- Manual inspection
- Team explanations
- Workflow consistency
- Domain plausibility
- Reproducibility
- Methodological discipline

---

## 3. What Should Be Evaluated

Instead of asking:

> Did the team prove the paradox perfectly?

The better question is:

> Was the team's workflow clean, systematic, reproducible, interpretable, and statistically reasonable?

This is more fair because teams may work on different paradoxes, themes, surveys, and datasets.

The evaluation should focus on:

- Dataset authenticity indicators
- Cleaning discipline
- Reproducibility
- Notebook structure
- Statistical reasoning indicators
- Visualization quality
- Documentation quality
- Manual explanation consistency

---

## 4. Required Submission Structure

Each team should submit a structured folder:

```text
team_name/
│
├── raw_data.csv
├── cleaned_data.csv
├── analysis.ipynb
├── report.pdf
└── presentation.pptx
```

Minimum required files:

- `raw_data.csv`
- `cleaned_data.csv`
- `analysis.ipynb`
- `report.pdf`

Optional but recommended:

- `presentation.pptx`
- `README.md`
- `requirements.txt`
- `data_dictionary.csv`

---

## 5. Detection Module 1: Duplicate and Near-Duplicate Response Detection

### Objective

Identify:

- Exact duplicate rows
- Near-identical answer patterns
- Mass-copied submissions
- Repeated survey responses with minor edits

### Exact Duplicate Detection

```python
import pandas as pd

df = pd.read_csv("responses.csv")

duplicates = df[df.duplicated()]
print("Number of exact duplicates:", len(duplicates))
print(duplicates)
```

### Duplicate Detection Based on Selected Columns

Useful when IDs, timestamps, or metadata differ, but actual answers are identical.

```python
subset_cols = ['Q1', 'Q2', 'Q3', 'Q4']

duplicates = df[df.duplicated(subset=subset_cols)]
print(duplicates)
```

### Similarity Score Between Responses

This detects near-duplicate submissions.

```python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import pandas as pd

encoded = pd.DataFrame()

for col in df.columns:
    le = LabelEncoder()
    encoded[col] = le.fit_transform(df[col].astype(str))

similarity = cosine_similarity(encoded)
threshold = 0.95

for i in range(len(similarity)):
    for j in range(i + 1, len(similarity)):
        if similarity[i][j] > threshold:
            print(f"Highly similar responses: {i} and {j}")
```

### Red Flags

- Large number of exact duplicate rows
- Many rows with similarity above `0.90` or `0.95`
- Multiple teams submitting nearly identical structures
- Same answer pattern repeated with only small metadata differences

---

## 6. Detection Module 2: Statistical Consistency Checks

### Objective

Detect:

- Unrealistically balanced data
- Suspiciously perfect subgroup trends
- Artificial distributions
- Mechanically smooth or symmetric numeric columns

### Standard Deviation Check

Synthetic datasets may show unnaturally low variation.

```python
numeric_cols = df.select_dtypes(include='number')
print(numeric_cols.std())
```

Very low variance across multiple variables can indicate fabrication or over-controlled generation.

### Distribution Visualization

```python
import matplotlib.pyplot as plt

numeric_cols.hist(figsize=(12, 8))
plt.show()
```

Look for:

- Overly smooth distributions
- Unrealistic symmetry
- Perfect subgroup separation
- Too-clean patterns for messy real-world data

### Correlation Matrix

Artificial datasets may produce unnaturally strong or overly neat correlations.

```python
import seaborn as sns
import matplotlib.pyplot as plt

corr = numeric_cols.corr()
sns.heatmap(corr, annot=True)
plt.show()
```

### Red Flags

- Perfect or near-perfect correlations without strong domain reason
- Too many variables showing clean linear relationships
- Suspiciously balanced groups
- No messy variation in real-world fields

---

## 7. Detection Module 3: Entropy and Response Variability Detection

### Objective

Identify:

- Excessively repetitive answers
- Low diversity patterns
- Mechanically generated responses
- Template-like answer distributions

### Shannon Entropy

Lower entropy may indicate templated or artificially generated responses.

```python
from scipy.stats import entropy
from collections import Counter


def calculate_entropy(column):
    counts = Counter(column)
    probs = [v / sum(counts.values()) for v in counts.values()]
    return entropy(probs)


for col in df.columns:
    print(col, calculate_entropy(df[col]))
```

### Red Flags

- Very low entropy across multiple columns
- Same options selected repeatedly
- Unrealistically uniform answer patterns
- Lack of natural randomness in subjective survey responses

---

## 8. Detection Module 4: Repetitive Text and AI-Generated Text Indicators

### Objective

Detect:

- Repeated wording
- AI-style templated language
- Unnatural consistency in free-text fields
- Similar open-ended answers across respondents

### Duplicate Free-Text Responses

```python
text_col = "Feedback"

duplicates = df[text_col].duplicated().sum()
print("Duplicate text responses:", duplicates)
```

### TF-IDF Similarity

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

texts = df[text_col].fillna("")

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(texts)

sim_matrix = cosine_similarity(tfidf)
threshold = 0.90

for i in range(len(sim_matrix)):
    for j in range(i + 1, len(sim_matrix)):
        if sim_matrix[i][j] > threshold:
            print(f"Similar text responses: {i} and {j}")
```

### Red Flags

- Multiple free-text answers using similar structure
- Repeated polished phrasing
- Lack of spelling variation or informal respondent behavior
- Too many complete, grammatically polished responses
- Similar sentence length and tone across many respondents

---

## 9. Detection Module 5: Sampling Distribution and Synthetic Balancing Detection

### Objective

Detect:

- Artificial subgroup equalization
- Forced paradox construction
- Unrealistically perfect balancing
- Too-neat cross-tabulated structures

### Crosstab Distribution Analysis

```python
pd.crosstab(df['Department'], df['Outcome'])
```

### Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Department', hue='Outcome', data=df)
plt.show()
```

### Red Flags

- Perfectly equal subgroup counts
- Unrealistic symmetry
- Suspiciously ideal distributions
- Subgroups designed too cleanly to produce a paradox
- No natural imbalance in real-world categories

---

## 10. Detection Module 6: Timestamp Pattern Analysis

### Objective

Detect:

- Bulk submissions
- Bot-like behavior
- Unrealistic collection speed
- Artificially generated survey entries

### Method

```python
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp')
df['time_diff'] = df['Timestamp'].diff()

print(df['time_diff'])
```

### Red Flags

- Many submissions within seconds
- Uniform time intervals
- Large sudden clusters
- All responses submitted in a very short window
- No realistic gaps between human survey interactions

---

## 11. Detection Module 7: Outlier and Pattern Detection

### Objective

Detect anomalous records and suspicious response patterns.

### Isolation Forest Method

```python
from sklearn.ensemble import IsolationForest

numeric_cols = df.select_dtypes(include='number')
numeric = numeric_cols.fillna(0)

model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(numeric)

anomalies = df[df['anomaly'] == -1]
print(anomalies)
```

### Red Flags

- Records that strongly differ from the rest of the dataset
- Artificially inserted edge cases
- Impossible or inconsistent values
- Suspicious clusters of anomalies

---

## 12. Manual Random Audit Verification

Automated checks are not enough. Teams should also be asked to explain their data collection process.

Suggested audit questions:

- How were respondents approached?
- Where were responses collected?
- What was the approximate time window of collection?
- What sampling methodology was followed?
- Why were these variables chosen?
- What cleaning steps were performed?
- Why were some records removed?
- How did the team verify respondent authenticity?
- What problems occurred during data collection?

Fabricated datasets often fail during explanation, even when the spreadsheet looks clean.

---

## 13. Recommended Combined Detection Pipeline

A practical baseline workflow:

```text
1. File structure validation
        ↓
2. Raw vs cleaned dataset comparison
        ↓
3. Exact duplicate detection
        ↓
4. Near-duplicate similarity detection
        ↓
5. Timestamp analysis
        ↓
6. Entropy and variability checks
        ↓
7. Correlation and distribution analysis
        ↓
8. Text similarity detection
        ↓
9. Isolation Forest anomaly detection
        ↓
10. Notebook structure evaluation
        ↓
11. Notebook reproducibility test
        ↓
12. Manual audit and questioning
```

---

## 14. Workflow Neatness Evaluation

This is one of the most automatable parts of the system.

### File Structure Validation

Check whether required files exist and follow naming conventions.

```python
import os

required_files = [
    "raw_data.csv",
    "cleaned_data.csv",
    "analysis.ipynb",
    "report.pdf"
]

for file_name in required_files:
    if os.path.exists(file_name):
        print(f"{file_name}: OK")
    else:
        print(f"{file_name}: Missing")
```

### Red Flags

- Missing raw dataset
- Missing cleaned dataset
- No analysis notebook
- Poor folder organization
- Random file names
- No reproducibility instructions

---

## 15. Dataset Cleanliness Evaluation

The goal is not to judge the dataset content alone. The goal is to judge whether the team cleaned the dataset properly.

### Missing Value Reduction

```python
raw_missing = raw.isnull().sum().sum()
clean_missing = clean.isnull().sum().sum()

improvement = raw_missing - clean_missing
print("Missing value reduction:", improvement)
```

### Duplicate Removal

```python
raw_dup = raw.duplicated().sum()
clean_dup = clean.duplicated().sum()

print("Raw duplicates:", raw_dup)
print("Cleaned duplicates:", clean_dup)
```

### Formatting Consistency

Check:

- Consistent datatypes
- Standardized categories
- Clean column names
- Proper encoding of missing values

```python
print(clean.dtypes)
```

### Impossible Value Detection

Example:

```python
clean[clean['Age'] < 0]
```

Other examples:

- Negative age
- Impossible marks
- Invalid dates
- Out-of-range ratings
- Categories not present in the questionnaire

---

## 16. Notebook and Analysis Neatness Evaluation

A notebook reveals whether the team actually worked systematically.

### Markdown Explanation Count

Good notebooks explain steps.

```python
import nbformat

nb = nbformat.read("analysis.ipynb", as_version=4)

markdown_cells = sum(
    1 for cell in nb.cells if cell.cell_type == 'markdown'
)

code_cells = sum(
    1 for cell in nb.cells if cell.cell_type == 'code'
)

ratio = markdown_cells / max(code_cells, 1)
print("Markdown-to-code ratio:", ratio)
```

### Execution Order Consistency

Messy notebooks often have chaotic execution order.

```python
exec_counts = [
    cell.execution_count
    for cell in nb.cells
    if cell.cell_type == 'code'
]

print(exec_counts)
```

Check for:

- Skipped numbering
- Repeated execution disorder
- Cells executed out of sequence
- Missing outputs

### Visualization Count

```python
plot_keywords = ['plt.', 'sns.', '.plot(']
plot_count = 0

for cell in nb.cells:
    if any(keyword in str(cell.source) for keyword in plot_keywords):
        plot_count += 1

print("Visualization count:", plot_count)
```

### Variable Naming Quality

Basic bad-name detection:

```python
bad_names = ['x', 'a', 'temp', 'test']
```

This should only be used as a weak signal, not a final judgment.

---

## 17. Reproducibility Evaluation

This is one of the strongest automated quality checks.

### Automatically Execute Notebook

```bash
jupyter nbconvert --execute analysis.ipynb
```

If the notebook crashes, the team should receive a penalty.

### Red Flags

- Notebook does not run
- Missing dependencies
- Hardcoded local paths
- Results cannot be regenerated
- Analysis depends on manual hidden steps
- Outputs in notebook do not match submitted report

---

## 18. Analytical Workflow Completeness

Instead of fully judging whether the paradox is correct, check whether essential analytical steps exist.

### Required Sections

Search notebook or report for these sections:

- Problem statement
- Data collection method
- Data cleaning
- Exploratory data analysis
- Subgroup analysis
- Interpretation
- Limitations
- Conclusion

### Regex/Keyword-Based Detection

```python
required_sections = [
    "problem statement",
    "data cleaning",
    "exploratory analysis",
    "subgroup analysis",
    "interpretation",
    "limitations"
]
```

---

## 19. Visualization Quality Evaluation

This can be partially automated.

### Minimum Visualization Count

```python
if plot_count < 2:
    score -= 5
```

### Visual Checks

Penalize:

- No plots
- Unreadable plots
- Excessive clutter
- No subgroup visualization
- No labels or titles
- Plots that do not support the argument

---

## 20. Statistical Reasoning Indicators

Correctness cannot be fully automated, but the presence of statistical reasoning can be detected.

### Keywords to Search For

```python
keywords = [
    'groupby',
    'corr',
    'crosstab',
    'regression',
    'contingency',
    'chi-square',
    'mean',
    'median',
    'standard deviation',
    'confidence interval'
]
```

### Positive Indicators

- Subgroup comparison
- Crosstab analysis
- Correlation analysis
- Regression or contingency analysis where relevant
- Discussion of limitations
- Interpretation tied to data
- Clear distinction between observation and conclusion

---

## 21. Suggested Auto-Evaluation Rubric

| Component | Automation Level | Suggested Weight |
|---|---:|---:|
| Dataset cleanliness | Fully automatable | 20 |
| Notebook structure | Fully automatable | 15 |
| Reproducibility | Fully automatable | 20 |
| Documentation quality | Mostly automatable | 10 |
| Visualization quality | Partially automatable | 10 |
| Statistical reasoning | Semi-automatable | 10 |
| Interpretation quality | Manual | 15 |
| **Total** |  | **100** |

---

## 22. Best Practical Strategy

### Fully Automate

- File structure validation
- Dataset cleanliness
- Missing value comparison
- Duplicate detection
- Timestamp analysis
- Entropy checks
- Reproducibility
- Formatting checks
- Notebook structure
- Workflow completeness

### Semi-Automate

- Statistical sophistication
- Subgroup analysis presence
- Visualization richness
- Text similarity
- Suspicious distribution patterns

### Manual Evaluation

- Conceptual reasoning
- Paradox understanding
- Interpretation depth
- Data collection credibility
- Explanation consistency
- Domain plausibility

---

## 23. What Should Not Be Compared Across Teams

Since teams may work on different paradoxes, themes, surveys, and datasets, the evaluation should not directly compare:

- Final outcomes
- Statistical significance alone
- Whether the paradox emerged perfectly
- Dataset size alone
- Complexity of chosen theme alone

These are unfair comparison points because different real-world data collection settings naturally produce different outcomes.

---

## 24. What Should Be Compared Across Teams

The fair comparison criteria are:

- Rigor
- Neatness
- Reproducibility
- Workflow quality
- Clarity
- Methodological discipline
- Dataset plausibility
- Documentation quality
- Ability to explain decisions

This makes the evaluation scalable, fair, and defensible.

---

## 25. Final Baseline Philosophy

This project should not behave like a magical AI detector. That would be technically weak and easy to challenge.

Instead, it should behave like a **forensic workflow evaluator**.

A strong system should answer:

- Is the dataset suspicious?
- Are there signs of duplication or artificial generation?
- Is the workflow reproducible?
- Did the team clean the data honestly?
- Does the analysis show methodological discipline?
- Can the team explain how the dataset was collected?

The final decision should be based on combined evidence, not one metric.

