# spam-filter
# 📩 Spam Message Detection

This is a simple **Machine Learning project** that detects whether a text message is **Spam** or **Ham (Not Spam)** using Python and scikit-learn.

---

## 🚀 Project Overview

- Uses the **Naive Bayes algorithm** for classification  
- Converts text messages into numeric features using **TF-IDF Vectorization**  
- Trains and tests the model on a **spam.csv** dataset  
- Predicts whether new messages are spam or not  

---

## 🧠 Steps in the Code

1. Load and clean the dataset (`spam.csv`)  
2. Encode the labels (ham/spam → 0/1)  
3. Split data into training and testing sets  
4. Convert text into TF-IDF features  
5. Train the **Multinomial Naive Bayes** model  
6. Evaluate performance using accuracy and reports  
7. Test the model on a new custom message  

---

## 📊 Example Output

Accuracy: 0.98
New message prediction: spam

---

## 📁 Requirements

Install the following Python libraries before running the script:
```bash
pip install pandas scikit-learn
▶️ How to Run
Download or clone this repository
Make sure the file spam.csv is in the same directory
Run the Python script:
python spam_detector.py
👨‍💻 Author
Prathamesh Nithyanandan
📧 prathamesh1524@gmail.com
🔗 https://github.com/PrathameshNithyanandan
