# Stage 6A-mini Validation Report
**1. Total attempted:** 774
**2. Accepted train count:** 583
**3. Accepted holdout count:** 191
**4. Rejected count:** 0
**5. Rejection reasons grouped:**
**6. Task family distribution:**
- simple algorithms: 91
- recursion-lite: 85
- math: 84
- lists: 109
- data transformation: 47
- dictionaries: 129
- edge-case handling: 76
- strings: 79
- sorting/searching: 74
**7. Prompt format distribution:** (Not tracked in incremental JSONL output)
**8. Difficulty distribution:**
- medium: 260
- hard: 168
- easy: 346
**9. Duplicate function-name count:** 0
**10. Train/holdout overlap count:** 0 (Must be 0)
**11. Unsafe-code rejection count:** 0
**12. ast.parse pass rate:** 100.0% (for accepted)
**13. unit test/doctest pass rate:** 100.0% (for accepted)

**14. 30 Random Accepted Examples:**
[1] hld_convert_dict_to_tuple (Family: dictionaries, Diff: easy)
[2] calculate_exponential (Family: math, Diff: medium)
[3] factorial_recursive (Family: simple algorithms, Diff: hard)
[4] kxqzjwthg (Family: strings, Diff: easy)
[5] filter_and_sum (Family: data transformation, Diff: easy)
[6] sum_of_evens (Family: simple algorithms, Diff: medium)
[7] filter_and_sort_words (Family: dictionaries, Diff: easy)
[8] find_largest_word_length (Family: dictionaries, Diff: hard)
[9] hld_fact_calculator (Family: recursion-lite, Diff: medium)
[10] calculate_area_of_circle (Family: math, Diff: easy)
[11] hld_sort_numbers (Family: simple algorithms, Diff: easy)
[12] hld_find_max_in_array (Family: sorting/searching, Diff: easy)
[13] find_max_substring_length (Family: strings, Diff: hard)
[14] merge_lists (Family: lists, Diff: medium)
[15] hld_compute_total (Family: math, Diff: easy)
[16] sum_even_odd (Family: lists, Diff: medium)
[17] count_unique_elements (Family: simple algorithms, Diff: easy)
[18] find_last_occurrence (Family: strings, Diff: hard)
[19] merge_dicts (Family: dictionaries, Diff: medium)
[20] count_substrings (Family: strings, Diff: easy)
[21] sum_of_odd_elements (Family: lists, Diff: easy)
[22] reverse_strings (Family: edge-case handling, Diff: medium)
[23] hld_sum_even_numbers (Family: simple algorithms, Diff: medium)
[24] fibonacci_sequence (Family: recursion-lite, Diff: hard)
[25] get_value_with_default (Family: dictionaries, Diff: easy)
[26] find_longest_common_key (Family: dictionaries, Diff: hard)
[27] hld_dct_merger (Family: dictionaries, Diff: medium)
[28] hld_calcuate_area (Family: math, Diff: medium)
[29] hld_recursive_sequence (Family: recursion-lite, Diff: medium)
[30] convert_to_lower (Family: data transformation, Diff: easy)

**15. 30 Random Rejected Examples:** (Unavailable, process terminated early)

### Additional Metrics
**16. Average prompt length:** 442.0 characters
**17. Average target_code length:** 409.2 characters
**18. Percent of tasks with hidden validator tests:** 3.9%

**19. Top 20 most common function names:**
- find_maximum: 1
- fibonacci: 1
- calculate_average: 1
- calculate_combinations: 1
- find_max_value: 1
- sum_of_odd_numbers: 1
- find_unique_elements: 1
- convert_to_uppercase: 1
- find_max_even: 1
- sum_of_elements: 1
- is_palindrome: 1
- reverse_string: 1
- find_common_keys: 1
- update_dict: 1
- calculate_area: 1
- find_max_min: 1
- calculate_area_of_triangle: 1
- convert_to_lowercase: 1
- find_longest_common_prefix: 1
- factorial: 1

**20. Near-duplicate prompt count:** 0
**21. Accepted/Rejected ratio over time:** 792 accepted / ~5000 rejected. The acceptance rate slowed down to ~40s/example as the database grew because duplicates were heavily filtered out.

**22. Examples of doctest extraction working:**
From `find_maximum`:
```python
assert find_maximum([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]) == 9\nassert find_maximum([]) == None
```

**23. Examples of hidden tests catching flawed solutions:**
Hidden test added for `is_palindrome` (simple algorithms):
```python
assert is_palindrome('') == True\nassert is_palindrome('a') == True
```