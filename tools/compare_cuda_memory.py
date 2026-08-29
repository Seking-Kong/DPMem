#!/usr/bin/env python
"""
Compare CUDA memory usage between two JSONL files.

Usage: python cudamem.py original.jsonl compressed.jsonl [-o output.jsonl]

Calculation rules:
- Sum max_allocated_mb across all devices in cuda_mem.
- If cuda_mem contains a peak key, use the data under peak.
- Calculate the percentage reduction from file1 to file2: (file1-file2)/file1*100.
"""

import argparse
import json
from pathlib import Path


def get_message_key(msg):
    """Extract the unique identifier for a message."""
    if "question_id" in msg and "videoID" in msg:
        return f"{msg['videoID']}_{msg['question_id']}"
    elif "question_id" in msg:
        return str(msg["question_id"])
    elif "id" in msg:
        return str(msg["id"])
    return None


def get_max_allocated_sum(cuda_mem):
    """Calculate the sum of max_allocated_mb across all devices."""
    if cuda_mem is None:
        return 0.0
    
    if isinstance(cuda_mem, dict) and "peak" in cuda_mem:
        devices = cuda_mem["peak"]
    elif isinstance(cuda_mem, list):
        devices = cuda_mem
    else:
        return 0.0
    
    return sum(d.get("max_allocated_mb", 0) for d in devices)


def load_jsonl(filepath):
    """Load a JSONL file and return {key: (msg, max_allocated_sum)}."""
    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            key = get_message_key(msg)
            if key:
                mem_sum = get_max_allocated_sum(msg.get("cuda_mem"))
                data[key] = {"msg": msg, "mem": mem_sum}
    return data


def compare_files(file1, file2, output_file=None):
    """Compare GPU memory usage between two files."""
    data1 = load_jsonl(file1)
    data2 = load_jsonl(file2)
    
    results = []
    matched_count = 0
    
    for key, info1 in data1.items():
        if key not in data2:
            continue
        
        matched_count += 1
        info2 = data2[key]
        mem1, mem2 = info1["mem"], info2["mem"]
        
        diff = mem1 - mem2
        reduce_pct = (mem1 - mem2) / mem1 * 100 if mem1 > 0 else 0.0
        
        results.append({
            "key": key,
            "mem_file1": round(mem1, 2),
            "mem_file2": round(mem2, 2),
            "diff_mb": round(diff, 2),
            "reduce_pct": round(reduce_pct, 2),
        })
    
    all_mem1 = [info["mem"] for info in data1.values()]
    all_mem2 = [info["mem"] for info in data2.values()]
    
    avg1 = sum(all_mem1) / len(all_mem1) if all_mem1 else 0
    avg2 = sum(all_mem2) / len(all_mem2) if all_mem2 else 0
    
    avg_reduce_pct = (avg1 - avg2) / avg1 * 100 if avg1 > 0 else 0
    
    summary = {
        "file1": str(file1),
        "file2": str(file2),
        "file1_count": len(data1),
        "file2_count": len(data2),
        "matched_count": matched_count,
        "avg_mem_file1_mb": round(avg1, 2),
        "avg_mem_file2_mb": round(avg2, 2),
        "avg_diff_mb": round(avg1 - avg2, 2),
        "avg_reduce_pct": round(avg_reduce_pct, 2),
    }
    
    if output_file:
        if output_file.endswith('.json'):
            output_file = output_file[:-5] + '.jsonl'
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + '\n')
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Results saved to: {output_file}")
    else:
        print(json.dumps(summary, ensure_ascii=False))
        for item in results:
            print(json.dumps(item, ensure_ascii=False))
    
    print(f"\nFile1: {file1} ({len(data1)} records)")
    print(f"File2: {file2} ({len(data2)} records)")
    print(f"Matched records: {matched_count}")
    print(f"File1 average GPU memory: {avg1:.2f} MB")
    print(f"File2 average GPU memory: {avg2:.2f} MB")
    print(f"Average difference: {avg1 - avg2:.2f} MB")
    print(f"Reduction from File1 to File2: {avg_reduce_pct:.2f}%")
    
    return {"summary": summary, "details": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare CUDA memory usage between two JSONL files"
    )
    parser.add_argument("file1", help="Path to the first JSONL file")
    parser.add_argument("file2", help="Path to the second JSONL file")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL path, generated automatically by default")
    
    args = parser.parse_args()
    
    if args.output:
        output_file = args.output
    else:
        name1 = Path(args.file1).stem
        name2 = Path(args.file2).stem
        output_file = f"result_{name1}_{name2}.jsonl"
    
    compare_files(args.file1, args.file2, output_file)
