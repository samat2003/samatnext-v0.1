"""
Generate a tiny synthetic smoke dataset matching the repo's expected JSONL format.
No real data needed — just valid Python prompt/target pairs for training stability checks.
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# 20 tiny Python code-gen examples in ChatML-compatible format
EXAMPLES = [
    {"prompt": "Write a Python function that adds two numbers.", "target": "def add(a, b):\n    return a + b"},
    {"prompt": "Write a Python function that multiplies two numbers.", "target": "def multiply(a, b):\n    return a * b"},
    {"prompt": "Write a Python function that returns the maximum of two numbers.", "target": "def maximum(a, b):\n    return a if a > b else b"},
    {"prompt": "Write a Python function that returns the minimum of two numbers.", "target": "def minimum(a, b):\n    return a if a < b else b"},
    {"prompt": "Write a Python function that computes the absolute value.", "target": "def absolute(x):\n    return x if x >= 0 else -x"},
    {"prompt": "Write a Python function that checks if a number is even.", "target": "def is_even(n):\n    return n % 2 == 0"},
    {"prompt": "Write a Python function that checks if a number is odd.", "target": "def is_odd(n):\n    return n % 2 != 0"},
    {"prompt": "Write a Python function that computes the square of a number.", "target": "def square(x):\n    return x * x"},
    {"prompt": "Write a Python function that computes the cube of a number.", "target": "def cube(x):\n    return x * x * x"},
    {"prompt": "Write a Python function that returns the length of a list.", "target": "def list_length(lst):\n    return len(lst)"},
    {"prompt": "Write a Python function that reverses a string.", "target": "def reverse_string(s):\n    return s[::-1]"},
    {"prompt": "Write a Python function that sums a list of numbers.", "target": "def sum_list(lst):\n    total = 0\n    for x in lst:\n        total += x\n    return total"},
    {"prompt": "Write a Python function that counts vowels in a string.", "target": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"},
    {"prompt": "Write a Python function that returns the factorial of n.", "target": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"},
    {"prompt": "Write a Python function that checks if a string is a palindrome.", "target": "def is_palindrome(s):\n    return s == s[::-1]"},
    {"prompt": "Write a Python function that returns the nth Fibonacci number.", "target": "def fibonacci(n):\n    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"},
    {"prompt": "Write a Python function that flattens a nested list.", "target": "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result"},
    {"prompt": "Write a Python function that removes duplicates from a list.", "target": "def remove_duplicates(lst):\n    seen = set()\n    result = []\n    for x in lst:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result"},
    {"prompt": "Write a Python function that sorts a list using bubble sort.", "target": "def bubble_sort(lst):\n    arr = lst[:]\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr"},
    {"prompt": "Write a Python function that computes the GCD of two numbers.", "target": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a"},
]

# Write train.jsonl (all 20 examples — tiny is fine for smoke)
train_path = os.path.join(OUT_DIR, "train.jsonl")
with open(train_path, "w", encoding="utf-8") as f:
    for ex in EXAMPLES:
        f.write(json.dumps(ex) + "\n")

print(f"Wrote {len(EXAMPLES)} examples to {train_path}")
print("Smoke dataset ready.")
