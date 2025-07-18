import json

def build_prompt(base_prompt, rlhf_data=None, ingestion_data=None, schema_data=None, example_data=None):
    messages = [{"role": "system", "content": base_prompt}]

    if rlhf_data:
        messages.append({
            "role": "user",
            "content": f"[RLHF Input JSON]\n{json.dumps(rlhf_data)[:1000]}..."
        })
    if ingestion_data:
        messages.append({
            "role": "user",
            "content": f"[Ingestion Mapping JSON]\n{json.dumps(ingestion_data)[:1000]}..."
        })
    if schema_data:
        messages.append({
            "role": "user",
            "content": f"[Output Schema JSON]\n{json.dumps(schema_data)[:1200]}..."
        })
    if example_data:
        messages.append({
            "role": "user",
            "content": f"[Expected Output JSON]\n{json.dumps(example_data)[:1200]}..."
        })

    return messages


def build_structure_prompt(base_prompt_text, rlhf_data):
    """
    Builds the explanation prompt and defers display of actual JSON to a downloadable file.
    """
    return [{"role": "system", "content": base_prompt_text}]


def get_first_two_tasks(rlhf_data):
    """
    Extracts and returns the first two full RLHF tasks as formatted JSON.
    """
    try:
        if not rlhf_data or "rlhf" not in rlhf_data:
            return json.dumps([], indent=2)

        tasks = rlhf_data["rlhf"][:2]
        return json.dumps(tasks, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to extract tasks: {str(e)}"})
