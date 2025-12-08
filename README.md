# Next Word Predictor using LSTM

This project implements a simple next-word prediction model using an LSTM (Long Short-Term Memory) neural network built with PyTorch.

## Project Overview

- **Goal**: Given an input text sequence, the model predicts the most likely next word.
- **Architecture**:
  - Tokenization and vocabulary building using NLTK.
  - Text sequences are converted to numerical indices and padded to a fixed length.
  - An embedding layer maps token indices to dense vectors.
  - An LSTM processes the embedded sequence and produces a hidden representation.
  - A fully connected layer maps the LSTM output to vocabulary logits for next-word prediction.
- **Core Technologies**:
  - Python
  - PyTorch (`torch`, `torch.nn`, `torch.optim`)
  - NLTK for tokenization
  - `torch.utils.data.Dataset` and `DataLoader` for batching

## Repository Structure

- `code.py` – (optional/script file, if used) additional code or experiments.
- `LSTM.ipynb` – main Jupyter Notebook containing data preparation, model definition, training loop, and prediction function.
- `requirements.txt` – Python dependencies required to run the notebook.

## How It Works

1. **Data Preparation**
   - A multi-line text document is used as the training corpus.
   - The text is lowercased and tokenized with NLTK's `word_tokenize`.
   - A vocabulary dictionary is built from token frequencies, with a special `<unk>` token for unknown words.
   - Sentences are split, converted to sequences of token indices, and expanded into training subsequences.
   - Sequences are padded to a common length and split into inputs (`x`) and targets (`y`), where `y` is the last word in each subsequence.

2. **Dataset and Dataloader**
   - A custom `Dataset` wraps the input and target tensors.
   - A `DataLoader` provides mini-batches for training.

3. **Model Architecture**
   - `Embedding`: maps vocabulary indices to 100-dimensional vectors.
   - `LSTM`: processes the sequence with hidden size 150.
   - `Linear` layer: projects the final hidden state to vocabulary size for next-word prediction.

4. **Training**
   - Loss: `CrossEntropyLoss`.
   - Optimizer: `Adam` with learning rate 0.001.
   - Training runs for a configurable number of epochs (e.g., 50), printing average loss per epoch.

5. **Prediction**
   - The `prediction(model, vocab, text)` function:
     - Tokenizes the input text.
     - Converts tokens to indices using the vocabulary (with `<unk>` for unseen tokens).
     - Pads the sequence to the maximum training length.
     - Feeds it through the trained model and selects the token with maximum logit as the predicted next word.

## Setup and Usage

1. **Create and activate a virtual environment** (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the notebook**:

```bash
jupyter notebook LSTM.ipynb
```

4. **Train and test**:
   - Execute the cells in `LSTM.ipynb` sequentially to:
     - Prepare the data.
     - Define and train the `LSTMModel`.
     - Call the `prediction` function to test next-word predictions, for example:

```python
prediction(model, vocab, "how much we will be covering the following ")
```

## Notes and Limitations

- The model is intentionally simple and trained on a relatively small, domain-specific text, so predictions are limited to the patterns seen in that corpus.
- This project is intended as an educational example of:
  - Text preprocessing and tokenization.
  - Building and training an LSTM for language modeling / next-word prediction.
  - Using PyTorch `Dataset`/`DataLoader` abstractions.

## Authorship and AI Assistance

- **Code & Logic**: All core code, design decisions, and implementation of the next-word prediction logic in this repository are **hand-written and created by the project author**.
- **README**: This `README.md` file is **AI-generated** based on the existing project structure and notebook, to document the project more clearly and consistently.

If you extend or modify the project, you may also want to update this README accordingly.