import sys
import json
import re
import argparse


def extract_first_json(text: str) -> dict:
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                raw = text[start:i + 1]
                return json.loads(raw)
    raise ValueError("Unclosed JSON object")


def clean_code(s: str) -> str:
    # Strip common markdown code fences first
    s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s, flags=re.MULTILINE)

    # Heuristically unwrap JSON-like wrappers of the form:
    # { "code": "..." } or { "completion": "..." }
    # Some models return invalid JSON with raw newlines inside the string,
    # so we cannot rely on json parsing here. Instead, peel off the wrapper
    # textually if it exists.
    s = s.strip()

    # Remove a leading { "code"|"completion": " prefix if present
    s = re.sub(r'^\{\s*"(?:code|completion)"\s*:\s*"', "", s, count=1, flags=re.DOTALL)
    # Remove a trailing " } if present
    s = re.sub(r'"\s*\}\s*$', "", s, count=1, flags=re.DOTALL)

    # Normalize/strip tokenizer artifacts and odd unicode from some model outputs
    # 1) Remove angle-bracketed special tokens like <|begin_of_sentence|>
    s = re.sub(r"<[^>]*>", "", s)
    # 2) Replace SentencePiece underscore U+2581 with a normal space
    s = s.replace("\u2581", " ")
    # 3) Normalize fullwidth vertical bar to ASCII and strip zero-width/control chars
    s = s.replace("\uff5c", "|")
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
    # 4) Collapse excessive internal whitespace introduced by cleaning
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract first JSON object and print cleaned 'code' field if present.")
    parser.add_argument("--input", "-i", default=None, help="Path to input file (default: read from stdin)")
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    obj = extract_first_json(text)
    code_value = obj.get("code") if isinstance(obj, dict) else None
    if code_value is None:
        code_value = obj.get("completion") if isinstance(obj, dict) else None
    if not isinstance(code_value, str):
        raise ValueError("JSON object does not contain a 'code' or 'completion' string field")
    cleaned = clean_code(code_value)
    # Print ONLY the cleaned code string
    sys.stdout.write(cleaned)


if __name__ == "__main__":
    main()


