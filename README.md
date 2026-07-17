# Automated-VQA-Dataset-Generation
An automated Visual Question Answer generation pipeline using Vision Language Models to create diverse multiple-choice QA pairs from images.

## Overview

This project presents an end-to-end pipeline for generating high-quality Visual Question Answering (VQA) datasets from images. The system leverages Vision-Language Models (VLMs) to understand image content, generate grounded descriptions, create diverse multiple-choice questions, and validate outputs before exporting them in a structured JSONL format.

The pipeline is designed to support both Hugging Face-hosted models and API-based multimodal models, making it extensible across different deployment environments. 

The pipeline follows a two-stage generation strategy:

1. Generate a grounded image description.
2. Generate category-specific multiple-choice VQA pairs from the description.

This design reduces hallucinations, improves question diversity, and enables easy integration with both Hugging Face and API-based multimodal models.
## Features

- Automated VQA generation
- Multiple question categories
- OCR
- Recognition
- Localization
- Counting
- Color
- Document Understanding
- Reasoning
- Hallucination filtering
- Diversity-aware question generation
- Balanced category scheduling
- JSONL dataset export
- Hugging Face and API model support

## Pipeline

1. **Image Loading**
2. **Image Understanding**
3. **Grounded Description Generation**
4. **Question Category Selection**
5. **MCQ Generation**
6. **Validation & Filtering**
7. **Dataset Export (JSONL)**
   
## Supported Models

### Hugging Face Models

- Qwen2.5-VL-7B-Instruct
- Qwen2.5-VL-3B-Instruct
- InternVL
- LLaVA

### API-Based Models

- GPT-4o
- Gemini
- Claude Vision

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- Accelerate
- Qwen-VL Utils
- Hugging Face Account (for model access)
- Google Colab Pro (recommended)
- NVIDIA GPU (A100/L4/T4 supported)

## Python Dependencies

```bash
pip install transformers accelerate torch torchvision qwen-vl-utils
```

## Installation

```bash
git clone https://github.com/debashish0404/Automated-VQA-Dataset-Generation.git

cd Automated-VQA-Dataset-Generation

pip install -r requirements.txt
```

## Project Structure

```text
Automated-VQA-Dataset-Generation/
│
├── README.md
├── report/
│   └── Pipeline_Report.pdf
│
├── images/
│   └── sample_images
│
├── output/
│   ├── vqa_dataset.jsonl
│   └── progress.json
│
└── src/
    └── vqa_generation_pipeline.py
```




## Sample Output

```json
{
  "id": "arc_0000008_q1",
  "image_id": "arc_0000008",
  "image_path": "/content/drive/MyDrive/interns/debashish/images/images/arc_0000008.jpg",
  "question": "Which of these best describes the content displayed on the computer screen?",
  "option_a": "A list of items with detailed descriptions.",
  "option_b": "A table with multiple rows and columns of data.",
  "option_c": "A map showing geographical locations.",
  "option_d": "A social media feed with user posts.",
  "correct_answer": "B",
  "qa_type": "recognition",
  "category": "recognition",
  "model_used": "Qwen/Qwen2.5-VL-7B-Instruct"
}
```
## Future Improvements

- Support for API-based VLMs (GPT-4o, Gemini, Claude)
- Multi-question generation per image
- Automatic quality scoring
- Human-in-the-loop validation
- Support for additional VQA categories
- Distributed large-scale generation

## License
This project is intended for research and dataset generation purposes.

## Author

**Debashish Mishra**  
Research Intern, IIT Gandhinagar

## Acknowledgements

Developed during a research internship at IIT Gandhinagar.

Special thanks to the research team for guidance and feedback during the development and evaluation of the VQA generation pipeline.


