# Drugs, Side Effects, and Medical Condition

## 📌 Project Overview
This project analyzes **drug effectiveness, side effects, and their correlation with medical conditions** using Python and data science techniques. The primary goal is to extract meaningful insights from a dataset containing patient reviews on various drugs and their experiences.

The project is implemented in **Google Colab** using a **Jupyter Notebook (.ipynb)** file, where data is loaded, cleaned, and analyzed to uncover trends related to drug effectiveness and side effects.

---

## 📂 Dataset Details
- **Dataset Source:** [Public Drug Review Dataset]
- **Attributes:**
  - `drugName`: Name of the drug.
  - `condition`: Medical condition the drug is prescribed for.
  - `review`: Patient-provided review of the drug.
  - `rating`: A numeric score (1-10) indicating effectiveness.
  - `sideEffects`: Reported side effects of the drug (if any).
  - `useDuration`: Duration of drug usage.

The dataset is **cleaned and preprocessed** to remove missing values, standardize text, and create structured formats suitable for analysis.

---

## 🛠️ Project Workflow & Code Explanation

### **1️⃣ Data Loading & Cleaning**
```python
import pandas as pd
import numpy as np

df = pd.read_csv('drug_reviews.csv')
df.dropna(inplace=True)  # Removing missing values
```
- **Pandas** is used for data manipulation.
- `dropna()` ensures only complete records are considered for accurate analysis.

### **2️⃣ Sentiment Analysis on Reviews**
Using **TextBlob**, the sentiment of each review is analyzed:
```python
from textblob import TextBlob

df['sentiment'] = df['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
```
- **Positive sentiment** (Polarity > 0) indicates a good drug experience.
- **Negative sentiment** (Polarity < 0) indicates a bad drug experience.

### **3️⃣ Most Commonly Used Drugs for Each Condition**
```python
most_common_drugs = df.groupby('condition')['drugName'].agg(lambda x: x.value_counts().idxmax())
```
- This groups the dataset by **medical condition** and finds the most frequently prescribed drug.

### **4️⃣ Analyzing Side Effects Frequency**
```python
side_effect_counts = df['sideEffects'].value_counts()
side_effect_counts[:10].plot(kind='bar')
```
- The top **10 most common side effects** are visualized using a bar chart.

---

## 📊 Findings & Insights
### ✅ **1. Drug Effectiveness Based on Ratings**
- Drugs with higher ratings (above **7**) had **positive sentiment** in patient reviews.
- Lower-rated drugs (**<4**) had a **high correlation with reported side effects**.
- Some drugs worked well for one condition but poorly for another.

### ✅ **2. Most Common Side Effects**
- **Nausea, Dizziness, and Fatigue** were the most frequently reported side effects.
- Drugs used for **mental health conditions** (depression, anxiety) had **higher reported side effects** than physical condition drugs.

### ✅ **3. Sentiment & Effectiveness Correlation**
- Patients who mentioned words like **"life-saving"**, "best drug ever" had a **high sentiment score (above 0.5)**.
- Reviews with words like **"worst experience"**, "severe reaction" had a **negative sentiment score (<-0.5)**.

### ✅ **4. Drug Duration vs. Effectiveness**
- Long-term drug users (6+ months) reported **higher satisfaction** than short-term users.
- Some drugs were **only effective after prolonged use**.

---

## 📌 Outputs & Visualizations
### **1️⃣ Top 10 Most Common Side Effects**
📊 **Bar Chart** showing the most reported side effects.

### **2️⃣ Drug Ratings Distribution**
📈 **Histogram** visualizing how ratings are distributed among different drugs.

### **3️⃣ Word Cloud for Positive & Negative Reviews**
📌 **Positive Reviews** highlight words like "effective," "life-saving," "best."
📌 **Negative Reviews** highlight words like "terrible," "severe reaction," "worst."

---

## 🏃‍♂️ How to Run the Notebook
### **1️⃣ Open in Google Colab**
1. Upload the `.ipynb` file to your Google Drive.
2. Open the notebook in **Google Colab**.

### **2️⃣ Install Required Libraries**
Run the following command in Colab:
```python
!pip install textblob wordcloud
```

### **3️⃣ Run Each Cell**
- Execute each cell step-by-step to see the analysis and visualizations.

---

## 🔥 Conclusion
This project provides **deep insights** into how different drugs perform for various conditions. By analyzing real-world patient reviews, we:
- Identified **effective drugs** for specific medical conditions.
- Found **common side effects** and their correlation with drug ratings.
- Explored the **relationship between sentiment & effectiveness**.

This analysis can be valuable for **doctors, pharmacists, and patients** in making informed decisions about medications.

---

## 🏆 Future Improvements
🚀 **Expand dataset** to include more drugs & conditions.
🚀 **Build a predictive model** for recommending drugs based on reviews.
🚀 **Enhance sentiment analysis** with deep learning (LSTM, BERT).

---

### ⭐ If you find this project helpful, give it a ⭐ on GitHub!

