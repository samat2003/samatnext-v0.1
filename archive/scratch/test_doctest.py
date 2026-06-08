def extract_doctests(code):
    tests = []
    lines = code.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('>>> '):
            expr = line[4:].strip()
            # Next line(s) are expected output until next >>> or end of docstring
            j = i + 1
            expected = []
            while j < len(lines) and not lines[j].strip().startswith('>>>') and lines[j].strip() not in ('"""', "'''") and not lines[j].strip().startswith('def '):
                if lines[j].strip():
                    expected.append(lines[j].strip())
                j += 1
            if expected:
                exp_str = '\n'.join(expected)
                tests.append(f"assert {expr} == {exp_str}")
            i = j
        else:
            i += 1
    return tests

code = '''
def add(a, b):
    """
    >>> add(1, 2)
    3
    >>> add(4, 5)
    9
    """
    return a + b
'''
print(extract_doctests(code))
