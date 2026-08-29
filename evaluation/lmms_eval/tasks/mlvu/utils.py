import os
import sys
from pathlib import Path

import yaml
from loguru import logger as eval_logger

hf_home = os.getenv("HF_HOME", "./~/.cache/huggingface")
base_cache_dir = os.path.expanduser(hf_home)


with open(Path(__file__).parent / "mlvu_dev.yaml", "r") as f:
    raw_data_dev = f.readlines()
    safe_data_dev = []
    for i, line in enumerate(raw_data_dev):
        if "!function" not in line:
            safe_data_dev.append(line)
    cache_name_dev = yaml.safe_load("".join(safe_data_dev))["dataset_kwargs"]["cache_dir"]
    cache_dir_dev = os.path.join(base_cache_dir, cache_name_dev)


with open(Path(__file__).parent / "mlvu_test.yaml", "r") as f:
    raw_data_test = f.readlines()
    safe_data_test = []
    for i, line in enumerate(raw_data_test):
        if "!function" not in line:
            safe_data_test.append(line)
    cache_name_test = yaml.safe_load("".join(safe_data_test))["dataset_kwargs"]["cache_dir"]
    cache_dir_test = os.path.join(base_cache_dir, cache_name_test)


# Mapping from question_type to numbered video subdirectory names.
QUESTION_TYPE_TO_DIR = {
    "plotQA": "1_plotQA",
    "findNeedle": "2_needle",
    "ego": "3_ego",
    "count": "4_count",
    "order": "5_order",
    "anomaly_reco": "6_anomaly_reco",
    "topic_reasoning": "7_topic_reasoning",
    "sub_scene": "8_sub_scene",
    "summary": "9_summary",
}

# Option-letter mapping.
CHOICE_LETTERS = ["A", "B", "C", "D"]


def mlvu_doc_to_visual_dev(doc):
    question_type = doc["question_type"]
    video_name = doc["video"]
    # Find the correct video subdirectory from question_type.
    video_dir = QUESTION_TYPE_TO_DIR.get(question_type, question_type)
    video_path = os.path.join(cache_dir_dev, "video", video_dir, video_name)
    if os.path.exists(video_path):
        return [video_path]
    # Fallback: use question_type directly as the directory name.
    video_path = os.path.join(cache_dir_dev, "video", question_type, video_name)
    if os.path.exists(video_path):
        return [video_path]
    sys.exit(f"video path:{video_path} does not exist, please check")


def mlvu_doc_to_visual_test(doc):
    video_path = doc["video_name"]
    video_path = os.path.join(cache_dir_test, video_path)
    if os.path.exists(video_path):
        return [video_path]
    sys.exit(f"video path:{video_path} does not exist, please check")


# def mlvu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
#     if lmms_eval_specific_kwargs is None:
#         lmms_eval_specific_kwargs = {}
#     question = doc["question"]
#     pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
#     post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
#     full_prompt = pre_prompt + question + post_prompt
#     return full_prompt


def mlvu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    question = doc["question"]
    candidates = doc.get("candidates", [])
    option_str = ""
    for i, cand in enumerate(candidates):
        if i < len(CHOICE_LETTERS):
            option_str += f"\n{CHOICE_LETTERS[i]}. {cand}"
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return pre_prompt + question + option_str + post_prompt


def extract_characters_regex(s):
    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

    return extract_mcq_answer(s, choices=CHOICE_LETTERS)


def answer_text_to_letter(answer_text, candidates):
    """将答案文本转换为选项字母(A/B/C/D)，若无candidates则返回原始答案文本"""
    if not candidates:
        return answer_text
    for i, candidate in enumerate(candidates):
        if candidate.strip() == answer_text.strip():
            return CHOICE_LETTERS[i] if i < len(CHOICE_LETTERS) else ""
    return ""


def mlvu_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: benchmark score
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)

    task_type = doc.get("task_type", doc.get("question_type", "unknown"))
    answer_text = doc["answer"]
    candidates = doc.get("candidates", [])

    # Convert the answer text to an option letter.
    answer_letter = answer_text_to_letter(answer_text, candidates)

    data_dict = {
        "question_id": doc["question"],
        "task_type": task_type,
        "pred_answer": pred_ans,
        "answer": answer_letter,
    }

    return {"mlvu_percetion_score": data_dict}


def mlvu_aggregate_results_dev(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    TASK_TYPES = {"anomaly_reco", "count", "ego", "findNeedle", "order", "plotQA", "topic_reasoning"}
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result["task_type"]
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    task_category_scores = {}

    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores[task_cate] = accuracy
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {accuracy:.1f}%")

    if TASK_TYPES:
        average_accuracy = sum(task_category_scores.values()) / len(TASK_TYPES)
    else:
        average_accuracy = 0

    eval_logger.info(f"Average Performance Across All Task Categories: {average_accuracy:.1f}%")

    return average_accuracy


def mlvu_aggregate_results_test(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    TASK_TYPES = {"anomaly_reco", "count", "ego", "findNeedle", "order", "plotQA", "topic_reasoning"}
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result["task_type"]
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    task_category_scores = {}

    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores[task_cate] = accuracy
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {accuracy:.1f}%")

    if TASK_TYPES:
        average_accuracy = sum(task_category_scores.values()) / len(TASK_TYPES)
    else:
        average_accuracy = 0

    eval_logger.info(f"Average Performance Across All Task Categories: {average_accuracy:.1f}%")

    return average_accuracy
