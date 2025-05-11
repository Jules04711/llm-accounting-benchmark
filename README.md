# LLM Benchmarking for Accounting

A Streamlit-based application that **benchmarks large language models (LLMs)** on accounting knowledge and evaluates their performance using structured, expert-led criteria.

This tool is designed for **accounting firms, educators, and AI practitioners** who want to assess the clarity, accuracy, and completeness of AI-generated responses to domain-specific prompts—**all in a secure, local environment** using Ollama.

## Overview

This application automatically generates accounting questions of varying difficulty levels, sends them to multiple LLMs via Ollama, and evaluates the responses using quantitative metrics. The results are stored in a SQLite database for analysis and comparison over time.

---

## Features

* **Benchmark multiple LLMs** running via Ollama
* **Evaluate AI model responses** with structured rubrics (accuracy, completeness, clarity)
* **Generate domain-specific accounting questions** with varying difficulty levels (Basic, Intermediate, Advanced)
* **Use a designated evaluator model** to score responses objectively
* **Track and compare model performance** over time using a local SQLite database
* **Visualize results** in dashboards, cards, and performance tables
* **Audit-ready storage** of all model responses and scoring details
* **Token usage analytics** with prompt and completion token tracking
* **Response time measurement** to evaluate model efficiency
* **Advanced performance analysis** using machine learning techniques:
  * Performance clustering to identify model tiers
  * Principal Component Analysis (PCA) for visualizing model relationships
  * Domain-specific performance scoring
  * Feature importance analysis for model strengths

---

## Requirements

* Python 3.8+
* [Streamlit](https://streamlit.io)
* [Ollama](https://ollama.com) running locally
* SQLite (installed with Python)
* Python packages (installed via requirements.txt):
  * streamlit
  * requests
  * pandas
  * scikit-learn
  * matplotlib
  * seaborn
  * numpy
  * tiktoken

---

## Setup

1. Clone this repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Start the Ollama server locally (default: `http://localhost:11434`)
4. Launch the app:

   ```bash
   streamlit run app.py
   ```

---

## Benchmark Prompts

The application uses a comprehensive set of accounting prompts organized by domain and difficulty level. The prompts are stored in `accounting_llm_benchmark_prompts.json` and cover the following domains:

### Domains

1. **GAAP Compliance**
   - Questions about accounting standards, principles, and financial reporting requirements
   - Example: "What is the purpose of GAAP in financial reporting?"

2. **Tax Code Knowledge**
   - Questions about tax regulations, deductions, and compliance
   - Example: "Explain the concept of taxable income."

3. **Financial Statement Analysis**
   - Questions about financial ratios, performance metrics, and statement interpretation
   - Example: "What is the formula for return on assets (ROA)?"

4. **Audit Procedures**
   - Questions about audit methodology, controls, and professional standards
   - Example: "Define materiality in auditing."

### Difficulty Levels

Each domain includes questions at three difficulty levels:

1. **Basic**
   - Fundamental concepts and definitions
   - Standard accounting practices
   - Core principles and rules

2. **Intermediate**
   - Complex scenarios and applications
   - Multi-step calculations
   - Regulatory requirements

3. **Advanced**
   - Complex analysis and evaluation
   - Professional judgment scenarios
   - Integration of multiple concepts

### Prompt Structure

Each prompt in the JSON file includes:
```json
{
    "id": "unique_identifier",
    "domain": "one_of_four_domains",
    "prompt": "actual_question_text",
    "difficulty": "Basic|Intermediate|Advanced",
    "ideal_answer": "model_answer_for_evaluation",
    "needs_improvement": false
}
```

The benchmark includes 100 carefully curated prompts (25 per domain) to ensure comprehensive coverage of accounting knowledge and skills.

---

## Evaluation Rubric

The application uses a structured evaluation rubric (`accounting_llm_eval_rubric.json`) to assess model responses across three key criteria:

### Accuracy (1-10 points)
- **1-2**: Factually incorrect or misleading; major conceptual errors or misalignment with standards (GAAP, IRS, etc.).
- **3-4**: Key ideas are partially correct but contain technical errors or unsupported conclusions.
- **5-6**: Mostly correct with minor inaccuracies or unclear assumptions; lacks citations or nuanced application.
- **7-8**: Technically accurate and well-grounded; reflects current standards and sound logic.
- **9-10**: Factually flawless; incorporates or mirrors authoritative guidance (e.g., ASC/IRC sections); demonstrates professional-level judgment.

### Completeness (1-10 points)
- **1-2**: Critically incomplete; misses core elements or misinterprets the question.
- **3-4**: Attempts to answer but leaves out key issues, edge cases, or needed assumptions.
- **5-6**: Covers main points but lacks depth, detail, or follow-through; reasoning is partial.
- **7-8**: Addresses the prompt thoroughly with awareness of scope, dependencies, and potential risks.
- **9-10**: Exhaustive and insightful; anticipates edge cases, adds value through illustrative examples, and applies multi-step reasoning.

### Clarity (1-10 points)
- **1-2**: Disorganized or incoherent; may use vague, generic, or contradictory language.
- **3-4**: Understandable but loosely structured or verbose; key ideas get lost in wording.
- **5-6**: Clear and readable, but may lack polish, logical transitions, or hierarchy of points.
- **7-8**: Logically organized with concise, technical language; suitable for internal memos or summaries.
- **9-10**: Highly professional tone and structure; mirrors how a CPA, auditor, or analyst would communicate in writing.

The total score for each response is calculated as the sum of these three criteria (maximum 30 points). This rubric ensures consistent, objective evaluation of responses across different models and questions.

---

## Usage

1. Open the app in your browser (Streamlit will provide a local URL)
2. From the sidebar:
   * Select which LLM model to benchmark
   * Choose an evaluator model 
3. Click **"Start Benchmark"** to generate accounting questions and initiate model evaluations
4. Review real-time scoring and comparisons
5. Analyze detailed model performance including feedback and score breakdowns

### Benchmark Process

The application works as follows:
1. Generates accounting questions across different domains and difficulty levels
2. Sends each question to the selected LLM
3. Evaluates responses based on accuracy, completeness, and clarity
4. Stores all data in the SQLite database
5. Presents real-time results and visualizations

### Scoring System

The application implements a comprehensive evaluation framework based on a technical paper on LLM Response Evaluation for Accounting and Auditing. The scoring system assesses responses across three key dimensions:

| Metric | Description |
|--------|-------------|
| Accuracy | Correctness of response aligned with GAAP, IFRS, and regulatory requirements |
| Completeness | Thoroughness in addressing all necessary aspects of an accounting query |
| Clarity | Intelligibility and straightforwardness of the explanation |

Each dimension is scored on a scale of 1-10, with detailed rubrics for each score level. The total score is the sum of all three dimensions (maximum 30 points).

---

## Admin Features

The application includes an admin interface that provides:

* **SQL Database Review**: Run custom queries on the benchmark database
* **ML Model Analysis**: Apply machine learning techniques to identify:
  * Overall best-performing models
  * Best models by domain
  * Performance clusters and tiers
  * Visualization of model relationships
* **Performance metrics**: Advanced scoring including response time and token efficiency

---

## Database Structure

The application uses SQLite with the following schema:

* **benchmarks**: Tracks benchmark sessions with timestamp and evaluator model
* **questions**: Stores all generated questions with difficulty level, domain, and correct answers
* **responses**: Contains all model responses and evaluation metrics

---

## Ideal Use Cases

* Accounting firms evaluating AI tools for client work
* AI engineers testing model fine-tuning in finance
* Educators comparing LLMs for accounting education
* Audit teams testing LLM compliance in explanation quality
* Researchers analyzing LLM performance patterns in specialized domains

---

## Technical Implementation

* Built with Streamlit for rapid UI development
* Uses sklearn for machine learning analyses
* Implements tiktoken for token counting
* Connects to Ollama API for local LLM inference
* Stores all data in SQLite for persistence and analysis

---

## License

MIT License

Copyright (c) 2024 LLM Accounting Benchmark

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Contributing

Contributions welcome! Please submit a pull request or open an issue to discuss proposed changes.
