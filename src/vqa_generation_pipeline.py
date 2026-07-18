"""
Automated VQA Dataset Generation Pipeline

This script generates multiple-choice Visual Question Answering (VQA)
samples from images using Vision-Language Models (VLMs).

Pipeline Stages:
1. Image Loading
2. Grounded Description Generation
3. Category Selection
4. Multiple-Choice Question Generation
5. Validation and Filtering
6. JSONL Dataset Export

Features:
- Two-stage generation strategy
- Category balancing
- Hallucination filtering
- Diversity-aware question generation
- Progress tracking across sessions
- JSONL dataset export

Model Used:
Qwen2.5-VL-7B-Instruct

Author: Debashish Mishra
"""
# ============================================================
# SETUP
# ============================================================
!pip install -q transformers accelerate qwen-vl-utils torch torchvision

import os, json, re, random, gc
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ============================================================
# CONFIG
# ============================================================
IMAGES_DIR   = "/content/drive/MyDrive/interns/debashish/images/images"
OUTPUT_PATH  = "/content/drive/MyDrive/interns/debashish/output/vqa_dataset.jsonl"
PROGRESS_PATH = "/content/drive/MyDrive/interns/debashish/output/progress.json"

MODEL_NAME  = "Qwen/Qwen2.5-VL-7B-Instruct"
START_INDEX = 0      
BATCH_SIZE  = 50

TARGET_DISTRIBUTION = {
    "ocr":                     0.20,
    "recognition":             0.20,
    "localization":            0.15,
    "document_understanding":  0.15,
    "counting":                0.10,
    "color":                   0.10,
    "reasoning":               0.10,
}

# Starter templates per category — used both to instruct the model
# and to check/reject outputs that don't diversify.
STARTER_TEMPLATES = {
    "ocr": ["Read the", "According to the text,", "What text appears",
            "Which word", "Transcribe the"],
    "recognition": ["Identify the", "Which of these", "Name the",
                    "What is depicted", "Who or what is shown"],
    "localization": ["Where is the", "Which position", "In which part of the image",
                      "Locate the", "What is positioned"],
    "document_understanding": ["What type of document", "What is the purpose of",
                                "Which section", "What does the document indicate",
                                "How is the document structured"],
    "counting": ["How many", "Count the number of", "What is the total number of",
                 "How many distinct"],
    "color": ["What color is", "Which color dominates", "What shade",
              "Identify the color of"],
    "reasoning": ["Why does", "What can be inferred", "What is the likely reason",
                  "Based on the image, what", "What conclusion"],
}

HALLUCINATION_PATTERNS = [
    r"\bcannot be determined\b", r"\bnot visible in the image\b",
    r"\bimage does not show\b", r"\bunable to determine\b",
    r"\bas an ai\b", r"\bi don't have access\b", r"\bno image provided\b",
]

random.seed(42)

# ============================================================
# MODEL LOAD (7B)
# ============================================================
print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model loaded.")

# ============================================================
# PROGRESS TRACKING (persists across Colab sessions)
# ============================================================
def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"counts": {c: 0 for c in TARGET_DISTRIBUTION}, "total": 0,
            "recent_starters": []}

def save_progress(progress):
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)

def pick_category(progress):
    """Weighted deficit scheduling: pick the category furthest below target."""
    total = max(progress["total"], 1)
    deficits = {}
    for cat, target_frac in TARGET_DISTRIBUTION.items():
        current_frac = progress["counts"].get(cat, 0) / total
        deficits[cat] = target_frac - current_frac
    # sample from top-3 deficit categories, weighted, to avoid rigid ordering
    ranked = sorted(deficits.items(), key=lambda x: x[1], reverse=True)[:3]
    cats, weights = zip(*ranked)
    weights = [max(w, 0.001) for w in weights]
    return random.choices(cats, weights=weights, k=1)[0]

# ============================================================
# STARTER DIVERSITY ENFORCEMENT
# ============================================================
def starter_of(question):
    return question.strip().split(",")[0].split(" ")[0:3]

def is_starter_overused(question, recent_starters, window=8, max_repeats=2):
    first_words = " ".join(question.strip().split()[:2]).lower()
    recent_first_words = [" ".join(s.strip().split()[:2]).lower()
                           for s in recent_starters[-window:]]
    return recent_first_words.count(first_words) >= max_repeats

# ============================================================
# HALLUCINATION FILTER
# ============================================================
def looks_hallucinated(record):
    blob = " ".join([
        record.get("question", ""),
        record.get("option_a", ""), record.get("option_b", ""),
        record.get("option_c", ""), record.get("option_d", ""),
    ]).lower()
    return any(re.search(p, blob) for p in HALLUCINATION_PATTERNS)

def looks_malformed(record):
    required = ["question", "option_a", "option_b", "option_c", "option_d", "correct_answer"]
    if any(not record.get(k) for k in required):
        return True
    opts = [record["option_a"], record["option_b"], record["option_c"], record["option_d"]]
    if len(set(o.strip().lower() for o in opts)) < 4:   # duplicate options
        return True
    if record["correct_answer"].upper() not in ("A", "B", "C", "D"):
        return True
    return False

# ============================================================
# MODEL CALL HELPER
# ============================================================
def run_qwen(image_path, prompt_text, max_new_tokens=512):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                        padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text.strip()

# ============================================================
# PASS 1: DESCRIPTION
# ============================================================
def describe_image(image_path):
    prompt = (
        "Describe this print-media image in detail. Cover: any visible text "
        "(headlines, captions, labels), objects/people present, layout/structure, "
        "colors, spatial arrangement, and any countable items. Be factual and "
        "only describe what is actually visible — do not guess or invent details."
    )
    return run_qwen(image_path, prompt, max_new_tokens=400)

# ============================================================
# PASS 2: MCQ GENERATION (conditioned on description + category)
# ============================================================
def build_pass2_prompt(description, category, avoid_starters):
    templates = STARTER_TEMPLATES[category]
    avoid_str = ", ".join(f'"{s}"' for s in avoid_starters[-5:]) if avoid_starters else "none"
    return f"""You are generating a multiple-choice VQA question for category "{category}".

Image description (grounded, factual — use ONLY this, do not invent details):
{description}

Requirements:
- The question MUST be about the "{category}" aspect of the image.
- The question must start differently from these recently used openers: {avoid_str}
- Prefer starting the question with one of: {', '.join(templates)}
- Do NOT start with a generic "What is shown" / "What type of" unless it fits naturally with the templates above.
- Provide exactly 4 options (A-D), only one correct, based strictly on the description.
- Do not reference "the description" — phrase it as if asked about the image directly.

Return ONLY valid JSON, no markdown, no commentary, in this exact schema:
{{
  "question": "...",
  "option_a": "...",
  "option_b": "...",
  "option_c": "...",
  "option_d": "...",
  "correct_answer": "A"
}}"""

def parse_json_response(raw_text):
    cleaned = re.sub(r"^```json|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def generate_mcq(image_path, description, category, recent_starters, max_retries=3):
    for attempt in range(max_retries):
        prompt = build_pass2_prompt(description, category, recent_starters)
        raw = run_qwen(image_path, prompt, max_new_tokens=300)
        parsed = parse_json_response(raw)
        if not parsed:
            continue
        if looks_malformed(parsed) or looks_hallucinated(parsed):
            continue
        if is_starter_overused(parsed["question"], recent_starters):
            continue  # force retry with fresh avoid-list context
        return parsed
    return None  # give up after retries; caller should skip this image/category

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    progress = load_progress()
    all_images = sorted(
        f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    batch = all_images[START_INDEX: START_INDEX + BATCH_SIZE]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_f = open(OUTPUT_PATH, "a")

    for i, fname in enumerate(batch):
        idx = START_INDEX + i
        image_id = os.path.splitext(fname)[0]
        image_path = os.path.join(IMAGES_DIR, fname)

        category = pick_category(progress)

        try:
            description = describe_image(image_path)
            mcq = generate_mcq(image_path, description, category, progress["recent_starters"])
        except Exception as e:
            print(f"[{idx}] ERROR on {fname}: {e}")
            continue

        if mcq is None:
            print(f"[{idx}] Skipped {fname} — failed to produce valid {category} question after retries")
            continue

        record = {
            "id": f"{image_id}_q1",
            "image_id": image_id,
            "image_path": image_path,
            "question": mcq["question"],
            "option_a": mcq["option_a"],
            "option_b": mcq["option_b"],
            "option_c": mcq["option_c"],
            "option_d": mcq["option_d"],
            "correct_answer": mcq["correct_answer"].upper(),
            "qa_type": category,          # <-- fixed: no longer hardcoded
            "category": category,
            "model_used": MODEL_NAME,
        }

        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

        progress["counts"][category] = progress["counts"].get(category, 0) + 1
        progress["total"] += 1
        progress["recent_starters"].append(mcq["question"])
        progress["recent_starters"] = progress["recent_starters"][-20:]
        save_progress(progress)

        print(f"[{idx}] {fname} -> [{category}] {mcq['question']}")

        if idx % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    out_f.close()
    print(f"\nDone. Next session set START_INDEX = {START_INDEX + len(batch)}")

if __name__ == "__main__":
    main()
