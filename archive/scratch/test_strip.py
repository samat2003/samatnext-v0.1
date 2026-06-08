import ast

def strip_fences(code):
    if code.startswith("```"):
        code = code.split("\n", 1)[-1]
        if code.endswith("```"):
            code = code[:-3]
    if code.startswith("python\n"):
        code = code[7:]
    return code.strip()

code_str = """```python
def remove_vowels(s: str) -> str:
    return s
```"""
stripped = strip_fences(code_str)
print("STRIPPED:\n" + stripped)
try:
    parsed = ast.parse(stripped)
    func_names = [node.name for node in ast.walk(parsed) if isinstance(node, ast.FunctionDef)]
    print("FUNCTIONS:", func_names)
except Exception as e:
    print("ERROR:", e)

