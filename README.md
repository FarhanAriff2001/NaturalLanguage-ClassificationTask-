# NaturalLanguage-ClassificationTask

## Overview
Designed and implemented a language detection system capable of identifying the language of a given sentence based on word usage and character-level patterns. Trained the model on diverse multilingual datasets and evaluated performance using leave-one-out cross-validation to improve generalization and accuracy. Iteratively refined feature selection and training methodology to enhance the model's overall success rate.

## Features
- Language detection from raw text input using word usage and character-level pattern analysis
- Naive Bayes classifier trained on multilingual datasets (general jokes and AI-themed jokes)
- Text preprocessing pipeline for cleaning and transforming raw input data
- Leave-one-out cross-validation for robust model evaluation and generalization
- Iterative feature selection and training refinement to maximize classification accuracy
- Output generation for classified results across multiple dataset types

## Tech Stack
- **Language:** Python
- **Frameworks / Libraries:** Naive Bayes (custom implementation)
- **Tools:** Custom preprocessing pipeline (`preprocess.py`), multilingual joke datasets as training/test corpora
