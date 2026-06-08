import re

def strip_fences(code):
    match = re.search(r'```(?:python)?\s*(.*?)```', code, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    if code.startswith("```"):
        code = code.split("\n", 1)[-1]
        if code.endswith("```"):
            code = code[:-3]
    if code.startswith("python\n"):
        code = code[7:]
    return code.strip()

print(strip_fences("Here is the code:\n```python\ndef foo():\n    pass\n```"))
