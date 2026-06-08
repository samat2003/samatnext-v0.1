"""
Stage 3 Paraphrase-Robust Execution Training — Data Generator
=============================================================
Architecture:
  33 tasks x 12 fn_name variants x 8 arg-name tuples x 20 paraphrase templates
  = ~63,360 theoretical maximum unique prompt strings

Guarantees:
  - All prompt strings are unique (strict dedup by exact string)
  - Train (50,000) and eval (500) sets are strictly disjoint
  - Fails fast with a clear error if pool < requested count
  - All targets validated: ast.parse + sandboxed unit test execution
  - Prints full diagnostics before saving files
"""
import os
import sys
import ast
import json
import random
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "stage3_paraphrase_train.jsonl")
EVAL_FILE  = os.path.join(DATA_DIR, "stage3_paraphrase_eval.jsonl")

NUM_TRAIN = 50_000
NUM_EVAL  = 500

SEED = 42

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def check_syntax(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def check_execution(code, tests):
    try:
        ns = {}
        exec(compile(code, "<string>", "exec"), ns)
        for t in tests:
            exec(compile(t, "<string>", "exec"), ns)
        return True
    except Exception:
        return False

def fn_defined(code, fn_name):
    try:
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                return True
    except Exception:
        pass
    return False

# ---------------------------------------------------------------------------
# Shared argument-name pools (8 tuples each)
# ---------------------------------------------------------------------------
POOL_2ARG_NUM = [
    ("a", "b"), ("x", "y"), ("num1", "num2"), ("val1", "val2"),
    ("left", "right"), ("first", "second"), ("n1", "n2"), ("p", "q"),
]
POOL_1ARG_NUM = [
    ("n",), ("x",), ("num",), ("val",), ("value",), ("number",), ("p",), ("count",),
]
POOL_1ARG_STR = [
    ("s",), ("string",), ("text",), ("msg",), ("word",), ("phrase",), ("line",), ("chars",),
]
POOL_1ARG_LST = [
    ("lst",), ("items",), ("arr",), ("array",), ("elements",), ("nums",), ("values",), ("data",),
]
POOL_2ARG_LST = [
    ("l1", "l2"), ("list1", "list2"), ("arr1", "arr2"), ("a", "b"),
    ("x", "y"), ("items1", "items2"), ("left", "right"), ("data1", "data2"),
]
POOL_2ARG_MIX = [
    ("d", "key"), ("data", "key"), ("mapping", "key"), ("dictionary", "k"),
    ("d", "k"), ("table", "key"), ("lookup", "k"), ("store", "key"),
]
POOL_2ARG_STR = [
    ("text", "substring"), ("s", "sub"), ("word", "substring"), ("content", "pattern"),
    ("text", "pattern"), ("haystack", "needle"), ("string", "sub"), ("text", "part"),
]

# ---------------------------------------------------------------------------
# 20-frame template generators (frame ACTION embeds {a}/{b} as needed)
# ---------------------------------------------------------------------------
def frames_2arg(action):
    """Return 20 unique paraphrase templates for a 2-argument task."""
    return [
        f"Write a Python function {{fn}}({{a}}, {{b}}) that {action}.",
        f"Create a function called {{fn}} that {action}, taking {{a}} and {{b}} as parameters.",
        f"Implement {{fn}}({{a}}, {{b}}) so that it {action}.",
        f"Define {{fn}} to accept {{a}} and {{b}} and {action}.",
        f"I need a Python function named {{fn}} that takes {{a}} and {{b}} and {action}.",
        f"Build a helper function {{fn}}({{a}}, {{b}}) that {action}.",
        f"Given inputs {{a}} and {{b}}, write a function called {{fn}} that {action}.",
        f"Please implement {{fn}}({{a}}, {{b}}); it should {action}.",
        f"Could you write {{fn}}({{a}}, {{b}})? It should {action}.",
        f"Write code for a function {{fn}} that accepts {{a}} and {{b}} and {action}.",
        f"I want a function {{fn}}({{a}}, {{b}}) that {action}.",
        f"Make a Python function named {{fn}} that accepts {{a}} and {{b}} and {action}.",
        f"Draft a function {{fn}}({{a}}, {{b}}) that {action}.",
        f"Develop {{fn}}({{a}}, {{b}}) to {action}.",
        f"Program a Python function {{fn}}({{a}}, {{b}}) that {action}.",
        f"The function {{fn}}({{a}}, {{b}}) should {action}.",
        f"I require a Python function named {{fn}} that accepts {{a}} and {{b}} and {action}.",
        f"Construct {{fn}}({{a}}, {{b}}) to {action}.",
        f"Provide a Python definition for {{fn}}({{a}}, {{b}}) that {action}.",
        f"Code up a function {{fn}}({{a}}, {{b}}) that {action}.",
    ]

def frames_1arg(action):
    """Return 20 unique paraphrase templates for a 1-argument task."""
    return [
        f"Write a Python function {{fn}}({{a}}) that {action}.",
        f"Create a function called {{fn}} that {action}, given {{a}}.",
        f"Implement {{fn}}({{a}}) so that it {action}.",
        f"Define {{fn}} to accept {{a}} and {action}.",
        f"I need a Python function named {{fn}} that takes {{a}} and {action}.",
        f"Build a helper function {{fn}}({{a}}) that {action}.",
        f"Given input {{a}}, write a function called {{fn}} that {action}.",
        f"Please implement {{fn}}({{a}}); it should {action}.",
        f"Could you write {{fn}}({{a}})? It should {action}.",
        f"Write code for a function {{fn}} that accepts {{a}} and {action}.",
        f"I want a function {{fn}}({{a}}) that {action}.",
        f"Make a Python function named {{fn}} that accepts {{a}} and {action}.",
        f"Draft a function {{fn}}({{a}}) that {action}.",
        f"Develop {{fn}}({{a}}) to {action}.",
        f"Program a Python function {{fn}}({{a}}) that {action}.",
        f"The function {{fn}}({{a}}) should {action}.",
        f"I require a Python function named {{fn}} that accepts {{a}} and {action}.",
        f"Construct {{fn}}({{a}}) to {action}.",
        f"Provide a Python definition for {{fn}}({{a}}) that {action}.",
        f"Code up a function {{fn}}({{a}}) that {action}.",
    ]

# ---------------------------------------------------------------------------
# Task definitions: 33 tasks, each with 12 fn_names + arg_pool + templates + code + tests
# ---------------------------------------------------------------------------
TASKS = [
    # ── ARITHMETIC 2-arg ────────────────────────────────────────────────────
    {
        "task_type": "add", "category": "arithmetic",
        "fn_names": ["add","sum_two","add_numbers","compute_sum","get_sum","addition","add_vals","calc_sum","plus","sum_nums","add_pair","total_two"],
        "arg_pool": POOL_2ARG_NUM,
        "templates": frames_2arg("returns the sum of {a} and {b}"),
        "code_template": "def {fn}({a}, {b}):\n    return {a} + {b}",
        "tests_fn": lambda fn, ag: [f"assert {fn}(2, 3) == 5", f"assert {fn}(-1, 1) == 0", f"assert {fn}(0, 0) == 0"],
    },
    {
        "task_type": "subtract", "category": "arithmetic",
        "fn_names": ["subtract","difference","sub","minus","sub_vals","get_diff","calc_diff","subtract_nums","deduct","compute_diff","take_away","diff_two"],
        "arg_pool": POOL_2ARG_NUM,
        "templates": frames_2arg("returns {a} minus {b}"),
        "code_template": "def {fn}({a}, {b}):\n    return {a} - {b}",
        "tests_fn": lambda fn, ag: [f"assert {fn}(5, 3) == 2", f"assert {fn}(1, 1) == 0", f"assert {fn}(-1, -2) == 1"],
    },
    {
        "task_type": "multiply", "category": "arithmetic",
        "fn_names": ["multiply","product","mul","times","get_product","calc_product","multiply_nums","compute_product","mult","product_of","times_two","mul_vals"],
        "arg_pool": POOL_2ARG_NUM,
        "templates": frames_2arg("returns the product of {a} and {b}"),
        "code_template": "def {fn}({a}, {b}):\n    return {a} * {b}",
        "tests_fn": lambda fn, ag: [f"assert {fn}(3, 4) == 12", f"assert {fn}(0, 5) == 0", f"assert {fn}(-2, 3) == -6"],
    },
    {
        "task_type": "divide", "category": "arithmetic",
        "fn_names": ["divide","quotient","div","calc_div","divide_nums","get_quotient","division","compute_div","divide_by","ratio","div_vals","fractional"],
        "arg_pool": POOL_2ARG_NUM,
        "templates": frames_2arg("returns {a} divided by {b}"),
        "code_template": "def {fn}({a}, {b}):\n    return {a} / {b}",
        "tests_fn": lambda fn, ag: [f"assert {fn}(10, 2) == 5.0", f"assert {fn}(1, 4) == 0.25", f"assert {fn}(9, 3) == 3.0"],
    },
    {
        "task_type": "power", "category": "arithmetic",
        "fn_names": ["power","exponentiate","raise_to","pow_val","get_power","calc_power","to_the_power","compute_power","raise_power","exp_val","power_of","exponent"],
        "arg_pool": POOL_2ARG_NUM,
        "templates": frames_2arg("returns {a} raised to the power of {b}"),
        "code_template": "def {fn}({a}, {b}):\n    return {a} ** {b}",
        "tests_fn": lambda fn, ag: [f"assert {fn}(2, 3) == 8", f"assert {fn}(5, 0) == 1", f"assert {fn}(3, 2) == 9"],
    },
    {
        "task_type": "modulo", "category": "arithmetic",
        "fn_names": ["modulo","remainder","mod_val","get_mod","calc_mod","mod_two","compute_mod","get_remainder","mod_result","modulus","rem_val","mod_nums"],
        "arg_pool": POOL_2ARG_NUM,
        "templates": frames_2arg("returns the remainder of {a} divided by {b}"),
        "code_template": "def {fn}({a}, {b}):\n    return {a} % {b}",
        "tests_fn": lambda fn, ag: [f"assert {fn}(10, 3) == 1", f"assert {fn}(8, 4) == 0", f"assert {fn}(7, 2) == 1"],
    },
    {
        "task_type": "maximum", "category": "arithmetic",
        "fn_names": ["maximum","get_max","max_val","find_max","larger","max_of_two","bigger","compute_max","max_result","pick_max","top_val","greatest"],
        "arg_pool": POOL_2ARG_NUM,
        "templates": frames_2arg("returns the larger of {a} and {b}"),
        "code_template": "def {fn}({a}, {b}):\n    return max({a}, {b})",
        "tests_fn": lambda fn, ag: [f"assert {fn}(3, 7) == 7", f"assert {fn}(-1, -5) == -1", f"assert {fn}(4, 4) == 4"],
    },
    {
        "task_type": "minimum", "category": "arithmetic",
        "fn_names": ["minimum","get_min","min_val","find_min","smaller","min_of_two","lesser","compute_min","min_result","pick_min","lowest_val","least"],
        "arg_pool": POOL_2ARG_NUM,
        "templates": frames_2arg("returns the smaller of {a} and {b}"),
        "code_template": "def {fn}({a}, {b}):\n    return min({a}, {b})",
        "tests_fn": lambda fn, ag: [f"assert {fn}(3, 7) == 3", f"assert {fn}(-1, -5) == -5", f"assert {fn}(4, 4) == 4"],
    },

    # ── ARITHMETIC 1-arg ────────────────────────────────────────────────────
    {
        "task_type": "absolute_value", "category": "arithmetic",
        "fn_names": ["absolute","abs_val","get_abs","absolute_value","calc_abs","math_abs","magnitude","abs_num","get_magnitude","nonneg","abs_result","strip_sign"],
        "arg_pool": POOL_1ARG_NUM,
        "templates": frames_1arg("returns the absolute value of {a}"),
        "code_template": "def {fn}({a}):\n    return abs({a})",
        "tests_fn": lambda fn, ag: [f"assert {fn}(-5) == 5", f"assert {fn}(3) == 3", f"assert {fn}(0) == 0"],
    },

    # ── BOOLEANS 1-arg ──────────────────────────────────────────────────────
    {
        "task_type": "is_even", "category": "booleans",
        "fn_names": ["is_even","check_even","even","even_check","number_is_even","detect_even","parity_even","even_number","num_is_even","test_even","even_val","verify_even"],
        "arg_pool": POOL_1ARG_NUM,
        "templates": frames_1arg("returns True if {a} is even, False otherwise"),
        "code_template": "def {fn}({a}):\n    return {a} % 2 == 0",
        "tests_fn": lambda fn, ag: [f"assert {fn}(4) is True", f"assert {fn}(3) is False", f"assert {fn}(0) is True"],
    },
    {
        "task_type": "is_odd", "category": "booleans",
        "fn_names": ["is_odd","check_odd","odd","odd_check","number_is_odd","detect_odd","parity_odd","odd_number","num_is_odd","test_odd","odd_val","verify_odd"],
        "arg_pool": POOL_1ARG_NUM,
        "templates": frames_1arg("returns True if {a} is odd, False otherwise"),
        "code_template": "def {fn}({a}):\n    return {a} % 2 != 0",
        "tests_fn": lambda fn, ag: [f"assert {fn}(3) is True", f"assert {fn}(4) is False", f"assert {fn}(0) is False"],
    },
    {
        "task_type": "is_positive", "category": "booleans",
        "fn_names": ["is_positive","check_positive","positive","is_gt_zero","above_zero","is_above_zero","detect_positive","positive_check","num_positive","test_positive","greater_zero","positive_val"],
        "arg_pool": POOL_1ARG_NUM,
        "templates": frames_1arg("returns True if {a} is strictly greater than zero, False otherwise"),
        "code_template": "def {fn}({a}):\n    return {a} > 0",
        "tests_fn": lambda fn, ag: [f"assert {fn}(5) is True", f"assert {fn}(-1) is False", f"assert {fn}(0) is False"],
    },

    # ── STRINGS 1-arg ────────────────────────────────────────────────────────
    {
        "task_type": "reverse_string", "category": "strings",
        "fn_names": ["reverse_string","rev_str","reverse","get_reversed","string_reverse","flip_string","backwards","invert_string","string_rev","rev_text","backwards_str","flip_str"],
        "arg_pool": POOL_1ARG_STR,
        "templates": frames_1arg("returns {a} with its characters in reverse order"),
        "code_template": "def {fn}({a}):\n    return {a}[::-1]",
        "tests_fn": lambda fn, ag: [f"assert {fn}('hello') == 'olleh'", f"assert {fn}('') == ''", f"assert {fn}('a') == 'a'"],
    },
    {
        "task_type": "to_lowercase", "category": "strings",
        "fn_names": ["to_lowercase","lower_str","make_lower","get_lower","lowercase","str_lower","downcase","all_lower","to_lower","lower_text","make_lowercase","lowercase_str"],
        "arg_pool": POOL_1ARG_STR,
        "templates": frames_1arg("returns {a} converted to all lowercase letters"),
        "code_template": "def {fn}({a}):\n    return {a}.lower()",
        "tests_fn": lambda fn, ag: [f"assert {fn}('HELLO') == 'hello'", f"assert {fn}('World') == 'world'", f"assert {fn}('abc') == 'abc'"],
    },
    {
        "task_type": "to_uppercase", "category": "strings",
        "fn_names": ["to_uppercase","upper_str","make_upper","get_upper","uppercase","str_upper","upcase","all_upper","to_upper","upper_text","make_uppercase","uppercase_str"],
        "arg_pool": POOL_1ARG_STR,
        "templates": frames_1arg("returns {a} converted to all uppercase letters"),
        "code_template": "def {fn}({a}):\n    return {a}.upper()",
        "tests_fn": lambda fn, ag: [f"assert {fn}('hello') == 'HELLO'", f"assert {fn}('World') == 'WORLD'", f"assert {fn}('ABC') == 'ABC'"],
    },
    {
        "task_type": "count_vowels", "category": "strings",
        "fn_names": ["count_vowels","vowel_count","num_vowels","get_vowel_count","find_vowels","tally_vowels","count_v","vowels_in","vowel_tally","total_vowels","vowel_num","get_vowels"],
        "arg_pool": POOL_1ARG_STR,
        "templates": frames_1arg("counts vowels (a, e, i, o, u) in {a} case-insensitively and returns the count"),
        "code_template": "def {fn}({a}):\n    return sum(1 for c in {a}.lower() if c in 'aeiou')",
        "tests_fn": lambda fn, ag: [f"assert {fn}('hello') == 2", f"assert {fn}('AEIOU') == 5", f"assert {fn}('xyz') == 0"],
    },
    {
        "task_type": "is_palindrome", "category": "strings",
        "fn_names": ["is_palindrome","palindrome_check","check_palindrome","is_pal","palindrome","pal_check","detect_palindrome","verify_palindrome","test_palindrome","palindrome_test","is_symmetric","symmetric_check"],
        "arg_pool": POOL_1ARG_STR,
        "templates": frames_1arg("returns True if {a} reads the same forwards and backwards, False otherwise"),
        "code_template": "def {fn}({a}):\n    return {a} == {a}[::-1]",
        "tests_fn": lambda fn, ag: [f"assert {fn}('racecar') is True", f"assert {fn}('hello') is False", f"assert {fn}('a') is True"],
    },
    {
        "task_type": "string_length", "category": "strings",
        "fn_names": ["string_length","str_len","get_length","length_of","len_str","count_chars","char_count","string_len","get_len","str_length","measure_str","text_length"],
        "arg_pool": POOL_1ARG_STR,
        "templates": frames_1arg("returns the number of characters in {a}"),
        "code_template": "def {fn}({a}):\n    return len({a})",
        "tests_fn": lambda fn, ag: [f"assert {fn}('hello') == 5", f"assert {fn}('') == 0", f"assert {fn}('abc') == 3"],
    },
    {
        "task_type": "strip_whitespace", "category": "strings",
        "fn_names": ["strip_whitespace","strip_str","trim","trim_str","strip_spaces","remove_whitespace","clean_str","strip_text","trim_text","remove_spaces","strip_ws","whitespace_strip"],
        "arg_pool": POOL_1ARG_STR,
        "templates": frames_1arg("returns {a} with leading and trailing whitespace removed"),
        "code_template": "def {fn}({a}):\n    return {a}.strip()",
        "tests_fn": lambda fn, ag: [f"assert {fn}('  hi  ') == 'hi'", f"assert {fn}('hello') == 'hello'", f"assert {fn}('  ') == ''"],
    },

    # ── STRINGS 2-arg ───────────────────────────────────────────────────────
    {
        "task_type": "contains_substring", "category": "strings",
        "fn_names": ["contains_substring","has_substring","is_in","string_contains","check_substring","substring_in","contains_str","has_str","find_in","text_contains","includes_str","string_has"],
        "arg_pool": POOL_2ARG_STR,
        "templates": frames_2arg("returns True if {b} is found within {a}, False otherwise"),
        "code_template": "def {fn}({a}, {b}):\n    return {b} in {a}",
        "tests_fn": lambda fn, ag: [f"assert {fn}('hello world', 'world') is True", f"assert {fn}('test', 'xyz') is False", f"assert {fn}('', 'a') is False"],
    },

    # ── LISTS 1-arg ─────────────────────────────────────────────────────────
    {
        "task_type": "safe_first", "category": "lists",
        "fn_names": ["safe_first","get_first_safe","first_or_none","head_safe","first_element","safe_head","get_first","first_safe","head_or_none","safe_get_first","peek_first","first_or_null"],
        "arg_pool": POOL_1ARG_LST,
        "templates": frames_1arg("returns the first element of {a}, or None if {a} is empty"),
        "code_template": "def {fn}({a}):\n    return {a}[0] if {a} else None",
        "tests_fn": lambda fn, ag: [f"assert {fn}([1, 2, 3]) == 1", f"assert {fn}([]) is None", f"assert {fn}(['x']) == 'x'"],
    },
    {
        "task_type": "safe_last", "category": "lists",
        "fn_names": ["safe_last","get_last_safe","last_or_none","tail_safe","last_element","safe_tail","get_last","last_safe","tail_or_none","safe_get_last","peek_last","last_or_null"],
        "arg_pool": POOL_1ARG_LST,
        "templates": frames_1arg("returns the last element of {a}, or None if {a} is empty"),
        "code_template": "def {fn}({a}):\n    return {a}[-1] if {a} else None",
        "tests_fn": lambda fn, ag: [f"assert {fn}([1, 2, 3]) == 3", f"assert {fn}([]) is None", f"assert {fn}(['x']) == 'x'"],
    },
    {
        "task_type": "sum_list", "category": "lists",
        "fn_names": ["sum_list","list_sum","get_sum","calculate_sum","total","sum_elements","add_all","total_list","list_total","sum_all","sum_items","aggregate"],
        "arg_pool": POOL_1ARG_LST,
        "templates": frames_1arg("returns the sum of all numbers in {a}"),
        "code_template": "def {fn}({a}):\n    return sum({a})",
        "tests_fn": lambda fn, ag: [f"assert {fn}([1, 2, 3]) == 6", f"assert {fn}([]) == 0", f"assert {fn}([-1, 1]) == 0"],
    },
    {
        "task_type": "max_list", "category": "lists",
        "fn_names": ["max_list","find_max","get_max","largest","list_max","maximum_item","list_maximum","max_element","biggest_item","top_element","get_largest","max_in_list"],
        "arg_pool": POOL_1ARG_LST,
        "templates": frames_1arg("returns the largest element in {a}, or None if {a} is empty"),
        "code_template": "def {fn}({a}):\n    return max({a}) if {a} else None",
        "tests_fn": lambda fn, ag: [f"assert {fn}([1, 5, 3]) == 5", f"assert {fn}([-1, -5]) == -1", f"assert {fn}([]) is None"],
    },
    {
        "task_type": "unique_preserve_order", "category": "lists",
        "fn_names": ["unique_preserve_order","dedup_ordered","remove_dups_stable","ordered_unique","stable_unique","unique_stable","dedup_stable","unique_in_order","preserve_unique","distinct_ordered","ndup_ordered","order_unique"],
        "arg_pool": POOL_1ARG_LST,
        "templates": frames_1arg("removes duplicates from {a} while preserving the original insertion order"),
        "code_template": "def {fn}({a}):\n    result = []\n    for item in {a}:\n        if item not in result:\n            result.append(item)\n    return result",
        "tests_fn": lambda fn, ag: [f"assert {fn}([1, 2, 1, 3]) == [1, 2, 3]", f"assert {fn}([]) == []", f"assert {fn}(['a', 'a', 'b']) == ['a', 'b']"],
    },
    {
        "task_type": "filter_positives", "category": "lists",
        "fn_names": ["filter_positives","keep_positives","only_positive","positive_only","get_positives","extract_positives","positives_only","select_positives","positive_filter","pos_only","filter_pos","retain_positives"],
        "arg_pool": POOL_1ARG_LST,
        "templates": frames_1arg("returns a list containing only the positive (greater than zero) elements from {a}"),
        "code_template": "def {fn}({a}):\n    return [x for x in {a} if x > 0]",
        "tests_fn": lambda fn, ag: [f"assert {fn}([-1, 2, 0, 3]) == [2, 3]", f"assert {fn}([]) == []", f"assert {fn}([-5, -1]) == []"],
    },
    {
        "task_type": "sum_evens", "category": "lists",
        "fn_names": ["sum_evens","add_even_numbers","total_evens","sum_even_only","even_sum","sum_of_evens","even_total","calc_even_sum","sum_even_nums","evens_total","get_even_sum","add_evens"],
        "arg_pool": POOL_1ARG_LST,
        "templates": frames_1arg("returns the sum of all even numbers in {a}"),
        "code_template": "def {fn}({a}):\n    return sum(x for x in {a} if x % 2 == 0)",
        "tests_fn": lambda fn, ag: [f"assert {fn}([1, 2, 3, 4]) == 6", f"assert {fn}([1, 3, 5]) == 0", f"assert {fn}([]) == 0"],
    },
    {
        "task_type": "flatten", "category": "lists",
        "fn_names": ["flatten","flatten_list","make_flat","flat_list","get_flattened","flatten_arr","one_level_flat","flatten_one","flat_one","reduce_nesting","flatten_once","un_nest"],
        "arg_pool": POOL_1ARG_LST,
        "templates": frames_1arg("flattens {a} by one level and returns a single list"),
        "code_template": "def {fn}({a}):\n    result = []\n    for sub in {a}:\n        result.extend(sub)\n    return result",
        "tests_fn": lambda fn, ag: [f"assert {fn}([[1, 2], [3]]) == [1, 2, 3]", f"assert {fn}([]) == []", f"assert {fn}([[], [1]]) == [1]"],
    },

    # ── LISTS 2-arg ─────────────────────────────────────────────────────────
    {
        "task_type": "merge_lists", "category": "lists",
        "fn_names": ["merge_lists","concat_lists","join_lists","combine_lists","append_lists","merge","join_arrays","cat_lists","combine","list_merge","merge_two","lists_combined"],
        "arg_pool": POOL_2ARG_LST,
        "templates": frames_2arg("returns {a} and {b} concatenated into a single list"),
        "code_template": "def {fn}({a}, {b}):\n    return {a} + {b}",
        "tests_fn": lambda fn, ag: [f"assert {fn}([1, 2], [3, 4]) == [1, 2, 3, 4]", f"assert {fn}([], [1]) == [1]", f"assert {fn}(['a'], ['b']) == ['a', 'b']"],
    },
    {
        "task_type": "merge_sorted", "category": "lists",
        "fn_names": ["merge_sorted","sorted_merge","combine_sorted","merge_and_sort","sort_merged","join_sorted","sorted_join","merge_sort_two","sorted_combine","combine_and_sort","merge_then_sort","sorted_union"],
        "arg_pool": POOL_2ARG_LST,
        "templates": frames_2arg("merges {a} and {b} and returns the combined result sorted in ascending order"),
        "code_template": "def {fn}({a}, {b}):\n    return sorted({a} + {b})",
        "tests_fn": lambda fn, ag: [f"assert {fn}([3, 1], [2, 4]) == [1, 2, 3, 4]", f"assert {fn}([], []) == []", f"assert {fn}([5], [1, 3]) == [1, 3, 5]"],
    },

    # ── DICTS ───────────────────────────────────────────────────────────────
    {
        "task_type": "safe_get", "category": "dicts",
        "fn_names": ["safe_get","dict_get","get_or_none","fetch_key","safe_lookup","get_key_safe","lookup_safe","dict_fetch","key_or_none","safe_key_get","get_or_default","dict_safe_get"],
        "arg_pool": POOL_2ARG_MIX,
        "templates": frames_2arg("returns the value for key {b} in dict {a}, or None if {b} is not present"),
        "code_template": "def {fn}({a}, {b}):\n    return {a}.get({b}, None)",
        "tests_fn": lambda fn, ag: [f"assert {fn}({{'x': 1}}, 'x') == 1", f"assert {fn}({{}}, 'missing') is None", f"assert {fn}({{'a': 'hello'}}, 'b') is None"],
    },
    {
        "task_type": "count_frequencies", "category": "dicts",
        "fn_names": ["count_frequencies","build_freq_map","tally_items","item_counts","frequency_map","count_each","freq_count","element_counts","tally","count_occurrences","build_counts","occurrence_map"],
        "arg_pool": POOL_1ARG_LST,
        "templates": frames_1arg("returns a dict mapping each unique element of {a} to its occurrence count"),
        "code_template": "def {fn}({a}):\n    counts = {{}}\n    for x in {a}:\n        counts[x] = counts.get(x, 0) + 1\n    return counts",
        "tests_fn": lambda fn, ag: [f"assert {fn}(['a', 'b', 'a']) == {{'a': 2, 'b': 1}}", f"assert {fn}([]) == {{}}", f"assert {fn}([1, 1, 1]) == {{1: 3}}"],
    },

    # ── ALGORITHMS ──────────────────────────────────────────────────────────
    {
        "task_type": "factorial", "category": "algorithms",
        "fn_names": ["factorial","compute_factorial","calc_fact","fact","get_factorial","n_factorial","factorialize","fact_of","compute_fact","calc_factorial","factorial_of","factorial_val"],
        "arg_pool": POOL_1ARG_NUM,
        "templates": frames_1arg("computes and returns the factorial of {a} (where factorial(0) = 1)"),
        "code_template": "def {fn}({a}):\n    if {a} == 0:\n        return 1\n    result = 1\n    for i in range(1, {a} + 1):\n        result *= i\n    return result",
        "tests_fn": lambda fn, ag: [f"assert {fn}(0) == 1", f"assert {fn}(5) == 120", f"assert {fn}(3) == 6"],
    },
    {
        "task_type": "fibonacci", "category": "algorithms",
        "fn_names": ["fibonacci","fib","nth_fib","fib_number","get_fib","compute_fib","fib_n","fib_val","calc_fib","fibonacci_of","fib_num","fib_at"],
        "arg_pool": POOL_1ARG_NUM,
        "templates": frames_1arg("returns the {a}-th Fibonacci number, where fib(0)=0 and fib(1)=1"),
        "code_template": "def {fn}({a}):\n    if {a} <= 0:\n        return 0\n    if {a} == 1:\n        return 1\n    a_, b_ = 0, 1\n    for _ in range(2, {a} + 1):\n        a_, b_ = b_, a_ + b_\n    return b_",
        "tests_fn": lambda fn, ag: [f"assert {fn}(0) == 0", f"assert {fn}(1) == 1", f"assert {fn}(6) == 8"],
    },
]

# ---------------------------------------------------------------------------
# Build and validate all examples
# ---------------------------------------------------------------------------
def render(template, fn, args):
    if len(args) == 1:
        return template.format(fn=fn, a=args[0])
    else:
        return template.format(fn=fn, a=args[0], b=args[1])

def make_record(uid, prompt, code, fn_name, args, tests, task):
    return {
        "id": f"stage3_{uid:06d}",
        "prompt": prompt,
        "target": code,
        "function_name": fn_name,
        "arguments": list(args),
        "tests": tests,
        "task_type": task["task_type"],
        "category": task["category"],
        "paraphrase_style": prompt.split('{')[0][:50].strip(),
        "difficulty": "easy",
    }

def build_all_examples():
    """Enumerate every (task × fn_name × arg_tuple × template) combination.
    Dedup by exact prompt string. Validate all targets."""
    seen_prompts = {}  # prompt_str -> record
    invalid_count = 0

    for task in TASKS:
        for fn_name in task["fn_names"]:
            for args in task["arg_pool"]:
                code = render(task["code_template"], fn_name, args)
                tests = task["tests_fn"](fn_name, args)

                # Validate reference solution once per (fn, args) combo
                if not check_syntax(code):
                    invalid_count += 1
                    continue
                if not fn_defined(code, fn_name):
                    invalid_count += 1
                    continue
                if not check_execution(code, tests):
                    invalid_count += 1
                    continue

                for tmpl in task["templates"]:
                    prompt = render(tmpl, fn_name, args)
                    if prompt not in seen_prompts:
                        uid = len(seen_prompts)
                        rec = make_record(uid, prompt, code, fn_name, args, tests, task)
                        seen_prompts[prompt] = rec

    if invalid_count > 0:
        print(f"[WARNING] {invalid_count} (fn, args) combos failed validation and were skipped.")

    return list(seen_prompts.values())

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 65)
    print("Stage 3 Paraphrase-Robust Dataset Generator")
    print("=" * 65)
    print(f"Tasks           : {len(TASKS)}")
    total_theoretical = sum(
        len(t["fn_names"]) * len(t["arg_pool"]) * len(t["templates"])
        for t in TASKS
    )
    print(f"Theoretical max : {total_theoretical:,} combinations")
    print(f"Requested train : {NUM_TRAIN:,}")
    print(f"Requested eval  : {NUM_EVAL:,}")
    print(f"Requested total : {NUM_TRAIN + NUM_EVAL:,}")
    print()

    print("Building and validating all combinations...")
    all_examples = build_all_examples()
    pool_size = len(all_examples)
    print(f"Actual unique prompts after dedup + validation: {pool_size:,}")

    # Fail fast
    needed = NUM_TRAIN + NUM_EVAL
    if pool_size < needed:
        print(f"\nERROR: Pool size ({pool_size:,}) is less than requested total ({needed:,}).")
        print(f"       Maximum possible: {pool_size:,} examples.")
        print("       Reduce NUM_TRAIN / NUM_EVAL or expand templates/variants.")
        sys.exit(1)

    print(f"Pool sufficient: {pool_size:,} >= {needed:,}. Proceeding.\n")

    # Shuffle and split
    random.shuffle(all_examples)
    eval_examples  = all_examples[:NUM_EVAL]
    train_examples = all_examples[NUM_EVAL:NUM_EVAL + NUM_TRAIN]

    # Verify disjoint
    eval_prompts  = {e["prompt"] for e in eval_examples}
    train_prompts = {e["prompt"] for e in train_examples}
    overlap = eval_prompts & train_prompts
    assert len(overlap) == 0, f"FATAL: {len(overlap)} prompts overlap between train and eval!"

    # ── Diagnostics ──────────────────────────────────────────────────────────
    print("-" * 65)
    print("DIAGNOSTICS")
    print("-" * 65)
    print(f"Train size         : {len(train_examples):,}")
    print(f"Eval size          : {len(eval_examples):,}")
    print(f"Train/eval overlap : 0 (verified)")

    task_dist = Counter(e["task_type"] for e in train_examples)
    print("\nTrain task distribution:")
    for task_type, cnt in sorted(task_dist.items()):
        print(f"  {task_type:<30} {cnt:>6,}")

    style_dist = Counter(e["prompt"].split()[0] for e in train_examples)
    print("\nTrain paraphrase first-word distribution (top 20):")
    for word, cnt in style_dist.most_common(20):
        print(f"  {word:<15} {cnt:>6,}")

    print("\nFirst 10 training examples:")
    for i, ex in enumerate(train_examples[:10], 1):
        print(f"\n[Train {i}]")
        print(f"  Task     : {ex['task_type']}")
        print(f"  Prompt   : {ex['prompt']}")
        print(f"  Target   :")
        for line in ex["target"].splitlines():
            print(f"    {line}")
        print(f"  Tests    : {ex['tests']}")

    print("\nFirst 5 eval examples:")
    for i, ex in enumerate(eval_examples[:5], 1):
        print(f"\n[Eval {i}]")
        print(f"  Task     : {ex['task_type']}")
        print(f"  Prompt   : {ex['prompt']}")
        print(f"  Target   :")
        for line in ex["target"].splitlines():
            print(f"    {line}")

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved {len(train_examples):,} training examples -> {TRAIN_FILE}")
    print(f"Saved {len(eval_examples):,} eval examples     -> {EVAL_FILE}")


if __name__ == "__main__":
    main()
