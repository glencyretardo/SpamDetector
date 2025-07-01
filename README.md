# Spam Detector Web Application

This project is a web-based **Spam Detection System** built using Python, Streamlit, and scikit-learn. It uses a supervised machine learning model—**Multinomial Naive Bayes**—to classify text messages as either **spam** or **ham** (not spam). The application allows users to input any message and receive an instant prediction of its classification.

> Developed by Julia Verzosa and Glency Retardo  
> BS Computer Science – University of Mindanao

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation & Usage](#installation--usage)
- [Application Structure](#application-structure)
- [Model Details](#model-details)
- [Sample Usage](#sample-usage)
- [License](#license)
- [Authors](#authors)

---

## Project Overview

The Spam Detector application demonstrates a practical application of Natural Language Processing (NLP) and Machine Learning in the field of spam classification. By training on a real-world dataset of SMS messages, this app provides real-time predictions via an interactive web interface built with Streamlit.

---

## Features

- ✅ Real-time prediction of spam or ham messages
- ✅ Interactive web interface using Streamlit
- ✅ Text preprocessing (lowercasing and vectorization)
- ✅ Classification using Multinomial Naive Bayes
- ✅ Optional display of accuracy score and classification report
- ✅ Clean and responsive user interface

---

## Technologies Used

| Technology      | Description                               |
|----------------|-------------------------------------------|
| Python 3.x      | Core programming language                |
| pandas          | Data loading and preprocessing           |
| scikit-learn    | Machine learning model and evaluation    |
| Streamlit       | Front-end web interface                  |
| CountVectorizer | Text vectorization                       |

---

## Dataset

The application uses the **SMS Spam Collection Dataset**, which contains a corpus of over 5,000 SMS messages tagged as either **spam** or **ham**.

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) / [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Format: CSV with columns `v1` (label) and `v2` (message text)

We renamed the columns to `label` and `text`, and all messages were converted to lowercase as part of the preprocessing.

---

## Installation & Usage

### Prerequisites

Ensure that you have Python installed (preferably version 3.8 or above). It is also recommended to use a virtual environment.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/spam-detector.git
cd spam-detector
