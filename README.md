# IMDB Sentiment Analysis with LSTM

## Overview

This project implements a sentiment analysis model using LSTM (Long Short-Term Memory) networks to classify IMDB movie reviews as positive or negative. The model is built with TensorFlow and Keras.

## Dataset

The project uses the IMDB dataset provided by Keras, which consists of 50,000 movie reviews split equally between training and testing sets. Each review is labeled as positive (1) or negative (0). Only the top 10,000 most frequent words are used in the vocabulary.

## Model Architecture

- **Embedding Layer**: Converts word indices to dense vectors of fixed size (32 dimensions)
- **Dropout Layer (0.2)**: Reduces overfitting after the embedding layer
- **LSTM Layer (32 units)**: Processes sequential data with recurrent dropout (0.2)
- **Dropout Layer (0.2)**: Further reduces overfitting
- **Dense Layer**: Single unit with sigmoid activation for binary classification

## Features

- Text sequence padding (500 tokens)
- Dropout for regularization
- Early stopping to prevent overfitting
- Performance visualization with accuracy and loss plots

## Requirements

- TensorFlow 2.x
- NumPy
- Matplotlib
- Jupyter Notebook

## How to Run

1. Open the `sentiment_analysis_task.ipynb` notebook in Jupyter
2. Run all cells in sequence
3. The model will automatically download the IMDB dataset, train, and evaluate
4. Performance metrics and visualizations will be displayed

## Results

The model achieves good performance on sentiment classification with regularization techniques to prevent overfitting:

- Early stopping monitors validation loss and stops training when performance plateaus
- Dropout layers reduce interdependent learning between neurons
- Performance metrics include accuracy and loss curves for both training and validation sets

## Future Improvements

- Try different network architectures (GRU, Bidirectional LSTM)
- Implement attention mechanisms
- Explore transfer learning with pre-trained language models
