# Midterm Exam: AI and ML for Cybersecurity
**Student:** Tornike Eristavi  
**Date:** January 9, 2026  
**GitHub Repository:** `aimlmid2026_t_eristavi25`

---

## 📊 Task 1: Correlation Analysis (10 Points)

### The Process
I extracted data points from the online graph provided at the midterm address. To understand the relationship between these points, I calculated the **Pearson Correlation Coefficient** using Python's `numpy` library.

* **Correlation Result:** **0.98**
* **Interpretation:** This high value indicates a **very strong positive linear relationship**. Essentially, as X increases, Y follows in a nearly perfect straight line.

### Visualization
The scatter plot below shows the blue data points and the red dashed trend line, confirming the 0.98 correlation.


* **Source Code:** [correlation/correlation_analysis.py](./correlation/correlation_analysis.py)
* **Graph:** ![Correlation Plot](./correlation/correlation_plot.png)

---

## 📧 Task 2: Spam Email Detection (20 Points)

### 1. Dataset & Source Code
* **Data File:** [t_eristavi25_92358.csv](./spam_classifier/t_eristavi25_92358.csv)
* **Application Code:** [spam_detector.py](./spam_classifier/spam_detector.py)

### 2. Model Development
I built a **Logistic Regression** model using `scikit-learn`. 
* **Training Strategy:** The data was split into **70% for training** and **30% for testing**.
* **Processing:** I used `pandas` to load four key features: word count, link count, capital letters, and spam keyword frequency.
* **Final Coefficients:** * `words`: 0.0065 | `links`: 0.8532 | `capital_words`: 0.4521 | `spam_word_count`: 0.8326
    * **Intercept:** -9.2909

### 3. Validation Results
Using the 30% test set, the model achieved the following:
* **Accuracy:** **96.27%**
* **Confusion Matrix:** 364 True Negatives and 358 True Positives.
    ```text
    [[364,   6],
     [ 22, 358]]
    ```


### 4. Manual Testing
I implemented a parser to test custom email text against the model.

* **Spam Test:** "WINNER! CLAIM YOUR FREE PRIZE NOW AT http://win.com http://prize.com"
    * **Result:** **SPAM**
    * **Why?** The high link count and all-caps text overcame the negative intercept.
* **Legit Test:** "Hey, are you coming to the office today? Let me know."
    * **Result:** **LEGITIMATE**
    * **Why?** No URLs or aggressive formatting keep the score below the threshold.

### 5. Visual Insights
* **Class Distribution:** ![Distribution](./spam_classifier/distribution.png)
    * *Insight:* Shows the dataset is balanced, ensuring the model isn't biased.
* **Heatmap:** ![Heatmap](./spam_classifier/heatmap.png)
    * *Insight:* Visually confirms that the model is highly accurate with very few errors.

---

## 🚀 How to Reproduce
1.  Install requirements: `pip install numpy pandas matplotlib seaborn scikit-learn`
2.  Run Correlation: `python3 correlation/correlation_analysis.py`
3.  Run Spam Detector: `python3 spam_classifier/spam_detector.py`
