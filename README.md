# Chat Models Battle Arena

A sophisticated framework for benchmarking and comparing different Large Language Models (LLMs) through interactive challenges and evaluations.

## Overview

This project implements a unique approach to benchmarking chat models by setting up a "battle arena" where:
- One model acts as a challenging quizmaster (Bot A)
- Another model acts as a student answering questions (Bot B)
- Two additional models serve as independent judges

Currently supported models include:
- DeepSeek (Reasoner)
- ChatGPT
- Claude
- Gemini
- xAI (Grok)

## Features

- **Interactive Evaluation**: Models engage in dynamic Q&A sessions
- **Multi-Judge System**: Two independent AI judges evaluate each interaction
- **Domain-Specific Testing**: Configurable domain focus for specialized evaluation
- **Comprehensive Metrics**: Evaluation of:
  - Question quality and difficulty
  - Answer accuracy and thoroughness
  - Overall performance tracking

## Prerequisites

- Python 3.8 or higher
- API keys for the following services:
  - OpenAI
  - Anthropic
  - Google AI (Gemini)
  - DeepSeek
  - xAI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Chat_models.git
cd Chat_models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables in a `.env` file:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
DEEPSEEK_API_KEY=your_deepseek_key
XAI_API_KEY=your_xai_key
```

## Usage

Run the benchmark:
```bash
python orchestration.py
```

The system will:
1. Initialize all models (quizmaster, student, and judges)
2. Conduct 5 rounds of Q&A
3. Generate comprehensive evaluations for each round
4. Provide a final summary from both judges

## Project Structure

```
Chat_models/
├── README.md
├── requirements.txt
├── .env                    # API keys and configuration
├── base.py                 # Base interface for chat models
├── orchestration.py        # Main orchestration logic
├── chatgpt_model.py       # ChatGPT implementation
├── claude_model.py        # Claude implementation
├── deepseek_model.py      # DeepSeek implementation
├── gemini_model.py        # Gemini implementation
├── xai_model.py           # xAI/Grok implementation
└── results/               # Output directory for benchmark results
```

## Output Format

Each round of evaluation includes:
1. Bot A's question
2. Bot B's answer with:
   - Difficulty rating (1-10)
   - Reasoning steps
   - Final answer
3. Two independent judge evaluations covering:
   - Question quality and challenge level
   - Answer accuracy and thoroughness
   - Critique and suggestions
   - Independent verification

## Contributing

Contributions are welcome! Areas for improvement include:
- Adding support for new models
- Implementing additional evaluation metrics
- Creating visualization tools for results
- Expanding domain-specific testing capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or feedback, please open an issue in the repository.