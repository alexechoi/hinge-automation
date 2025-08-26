# app/data_store.py

import json
import os
from datetime import datetime

DATA_FILE = "generated_comments.json"
FEEDBACK_FILE = "feedback_records.json"


def store_generated_comment(comment_id, profile_text, generated_comment, style_used):
    """
    Store details about each generated comment for future analysis.
    """
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "comment_id": comment_id,
        "profile_text": profile_text,
        "generated_comment": generated_comment,
        "style_used": style_used,
    }

    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w") as f:
            json.dump([], f)

    with open(DATA_FILE, "r+") as f:
        data = json.load(f)
        data.append(record)
        f.seek(0)
        json.dump(data, f, indent=2)


def calculate_template_success_rates():
    """
    Merge data from generated_comments.json and feedback_records.json
    to see which style is leading to the most matches.
    """
    if not (os.path.exists(DATA_FILE) and os.path.exists(FEEDBACK_FILE)):
        print("No data to calculate success rates.")
        return {}

    with open(DATA_FILE, "r") as f:
        comments_data = json.load(f)
    with open(FEEDBACK_FILE, "r") as f:
        feedback_data = json.load(f)

    # comment_id -> style
    comment_style_map = {}
    for c in comments_data:
        cid = c["comment_id"]
        style = c["style_used"]
        comment_style_map[cid] = style

    # Tally outcomes: style -> {matches, total}
    results = {}
    for fb in feedback_data:
        cid = fb["comment_id"]
        outcome = fb["outcome"]
        style = comment_style_map.get(cid)
        if not style:
            continue

        if style not in results:
            results[style] = {"matches": 0, "total": 0}
        results[style]["total"] += 1
        if outcome == "match":
            results[style]["matches"] += 1

    # Convert to success rates
    success_rates = {}
    for style, counts in results.items():
        if counts["total"] > 0:
            success_rates[style] = counts["matches"] / counts["total"]
        else:
            success_rates[style] = 0.0

    return success_rates
