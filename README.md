# Shakespeare RNN

This project implements a character-level Recurrent Neural Network (RNN) trained on Shakespeare's works. The model can generate Shakespeare-like text based on a given prompt.

## Table of Contents

- [Shakespeare RNN](#shakespeare-rnn)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
    - [Training](#training)
    - [Inference](#inference)
  - [Model Architecture](#model-architecture)
  - [Dataset](#dataset)
  - [Configuration](#configuration)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)

## Project Overview

This project uses a Long Short-Term Memory (LSTM) network to generate text in the style of Shakespeare. The model is trained on a dataset of Shakespeare's works and can generate new text based on a given prompt.

Key features:
- Character-level text generation
- LSTM-based RNN architecture
- Customizable hyperparameters
- Training with Weights & Biases logging
- Interactive inference script

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/shakespeare-rnn.git
   cd shakespeare-rnn
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

```
shakespeare-rnn/
│
├── data/
│   ├── __init__.py
│   └── dataset.py
│
├── model/
│   ├── __init__.py
│   └── rnn.py
│
├── utils/
│   ├── __init__.py
│   └── tokenizer.py
│
├── config.py
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```

## Usage

### Training

To train the model, run:

```
python train.py
```

This will start the training process and log the results to Weights & Biases. You can monitor the training progress in real-time through the W&B dashboard.

### Inference

To generate text using the trained model, run:

```
python inference.py
```

This will load the trained model and allow you to enter prompts for text generation. The script will also generate text for a few predefined prompts.

## Model Architecture

The model uses a character-level LSTM network with the following architecture:
- Embedding layer
- LSTM layer(s)
- Fully connected output layer

The exact architecture (number of layers, hidden dimensions, etc.) can be configured in the `config.py` file.

## Dataset

The model is trained on the Tiny Shakespeare dataset, which is a collection of Shakespeare's works. The dataset is automatically downloaded using the Hugging Face `datasets` library.

## Configuration

You can modify the model's hyperparameters and training settings in the `config.py` file. Key configurations include:
- Batch size
- Sequence length
- Embedding dimension
- Hidden dimension
- Number of LSTM layers
- Learning rate
- Number of training epochs

## Results

After training, you can find the training logs and performance metrics on the Weights & Biases dashboard. The trained model will be saved as `shakespeare_model.pth`, and the tokenizer will be saved as `tokenizer.pkl`.

Example generated text:

[Include some example outputs from your trained model here]

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature-name`)
6. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This README provides a comprehensive overview of your project, including installation instructions, usage guidelines, project structure, and other relevant information. You may want to customize some parts, such as the repository URL, example outputs, and any specific instructions or results from your implementation.

Remember to create a LICENSE file if you haven't already, and you may want to add more specific details about the performance of your model or any interesting findings from your experiments.