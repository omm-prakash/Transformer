# Transformer
This repository contains an implementation of a simplified version of the Transformer model, as described in the seminal paper "Attention Is All You Need" by Vaswani et al. The Transformer model revolutionized the field of natural language processing (NLP) by introducing the attention mechanism as a fundamental building block for sequence-to-sequence tasks.

## Overview

The Transformer model architecture has become a cornerstone in various NLP applications, including machine translation, text summarization, and language understanding tasks. This implementation aims to provide a clear and concise version of the Transformer model, making it accessible for educational purposes and experimentation.
The code uses the [Hugging Face](https://huggingface.co/docs/transformers/en/main_classes/tokenizer) tokenizer for the token handling. 

### Features
1. **Self-Attention Mechanism**: The core of the Transformer model, self-attention allows the model to weigh the importance of different input tokens dynamically.
2. **Multi-Head Attention**: The model utilizes multiple attention heads to capture different aspects of the input sequence simultaneously.
3. **Positional Encoding**: To provide positional information to the model, sinusoidal positional encodings are added to the input embeddings.
4. **Encoder-Decoder Architecture**: The Transformer model can be used for both encoder and decoder tasks, making it versatile for various NLP tasks such as machine translation and text summarization.
5. **Scaled Dot-Product Attention**: The attention mechanism used in the Transformer employs scaled dot-product attention to mitigate issues related to large input dimensions.

## Environment Setup

To recreate the environment required to run the code, you can use the provided `environment.yml` file with Conda. Simply execute the following command:

```bash
conda env create -f environment.yml
```

This will create a Conda environment with all the necessary dependencies installed.

## Training

The training script `train.py` contains the code for training the simplified Transformer model. You can run the training script using Python:

```bash
python train.py
```

All other files, alnogs with variables in `config.yml` are self-explanatory. Hence I am not making this document long!! 

Feel free to customize the training script according to your specific task and dataset requirements. Additionally, you can adjust hyperparameters, modify the model architecture, or incorporate different optimization techniques as needed.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al. (2017)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Official documentation for the PyTorch library

## Contributing

Contributions to this repository are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.
You may contact me at ommprakash2568@gmail.com.

Thank You,<br>
Omm Prakash Sahoo

