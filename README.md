# Sentiment_Analysis
# Restaurant Review Sentiment Analysis

## Overview

This project builds a machine learning model to automatically classify restaurant reviews as positive or negative, enabling efficient sentiment analysis for large volumes of customer feedback. By leveraging natural language processing (NLP) and machine learning techniques, the tool helps restaurant owners and stakeholders gain actionable insights to improve customer satisfaction and business performance.

## Features

- Automated classification of restaurant reviews into positive or negative sentiment.
- Text preprocessing including cleaning, tokenization, and normalization.
- Feature extraction using Bag-of-Words (BoW) and TF-IDF techniques.
- Model training and evaluation using classification algorithms.
- Achieved an accuracy of **79.91%** in sentiment prediction[5].
- User-friendly interface for real-time review sentiment prediction.

## Project Structure

- **Data Collection**: Reviews are sourced from platforms like Yelp and Google Reviews, and labeled for sentiment.
- **Preprocessing**: The text is cleaned, tokenized, and transformed into numerical features using BoW/TF-IDF.
- **Model Building**: Machine learning models such as Logistic Regression and Multinomial Naive Bayes are trained on the processed data.
- **Evaluation**: Model performance is assessed using accuracy, precision, recall, and F1-score metrics.
- **Deployment**: The trained model can be deployed as an API or integrated into a dashboard for real-time analysis[4][5].


## Results

- The sentiment analysis model achieved an accuracy of **79.91%** on the test dataset[5].
- Balanced precision and recall indicate robust performance in classifying both positive and negative reviews.

## Applications

- Track and visualize customer sentiment trends over time.
- Identify strengths and weaknesses in restaurant offerings.
- Support marketing and customer service decisions with data-driven insights.

## Requirements

- Python 3.x
- pandas, numpy, scikit-learn, nltk
- Jupyter Notebook

## Acknowledgments

- Datasets from Yelp, Google Reviews, and TripAdvisor.
- Libraries: scikit-learn, nltk, pandas, numpy.
