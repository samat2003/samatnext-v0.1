# Stage 6C Hugging Face Dataset Inspection Report (v2 - Strict Filtering)

## Overview
- **Dataset:** `jon-tow/starcoderdata-python-edu` (train split, streaming)
- **Rows Processed:** 31489
- **Accepted Examples:** 100
- **Overall Acceptance Rate:** 0.32%
- **Parse Rate (among size-filtered):** 10.33%

## Exact Filtering Rules Used (v2)
1. **Length:** ≤ 1500 characters (fits 512 context easily).
2. **Structure:** Exactly 1 top-level function. No classes, no async, no decorators.
3. **Execution:** No top-level calls. No `__main__` or script execution logic.
4. **Imports:** NO external imports. Allowed imports ONLY: `math, re, itertools, functools, collections, string, typing`.
5. **Side Effects:** NO `print()`, NO `input()`, NO `open()`, NO `read()`, NO `write()`.
6. **Tests:** NO `assert` statements inside function. No pytest/unittest boilerplate.
7. **Return:** Function must contain a `return` statement.
8. **Lines:** Body must be between 3 and 60 lines.
9. **Banned Ecosystems:** ML (torch, pandas, sklearn, numpy), Web (django, requests, flask), OS/Pathing (os, pathlib).

## Top Rejection Reasons
- **Too long (>1500 chars)**: 18505
- **Contains banned keyword**: 6299
- **SyntaxError/ParseFail**: 3433
- **Disallowed importFrom**: 1437
- **Disallowed import**: 630
- **Contains 0 functions (need exactly 1)**: 549
- **Contains class definition**: 312
- **Contains 2 functions (need exactly 1)**: 66
- **Contains top-level execution code**: 59
- **Contains print() statements**: 26
- **No return statement (function must return a value)**: 24
- **Contains 3 functions (need exactly 1)**: 15
- **Contains 4 functions (need exactly 1)**: 11
- **Function body too small (< 3 lines)**: 5
- **Contains 5 functions (need exactly 1)**: 5

## 50 Accepted Examples After Filtering

### Accepted 1
```python
def indexof(listofnames, value):
    if value in listofnames:
        value_index = listofnames.index(value)
        return(listofnames, value_index)
    else: return(-1)
```

### Accepted 2
```python
def ips_between(start, end):
    calc = lambda n, m: (int(end.split(".")[n]) - int(start.split(".")[n])) * m
    return calc(0, 256 * 256 * 256) + calc(1, 256 * 256) + calc(2, 256) + calc(3, 1)
```

### Accepted 3
```python
"""
Minimum edit distance computes the cost it takes to get from one string to another string. 
This implementation uses the Levenshtein distance with a cost of 1 for insertions or deletions and a cost of 2 for substitutions.

Resource: https://en.wikipedia.org/wiki/Edit_distance

For example, getting from "intention" to "execution" is a cost of 8.
minimum_edit_distance("intention", "execution")
# 8 
"""
def minimum_edit_distance(source, target):
    n = len(source)
    m = len(target)
    D = {}

    # Initialization
    for i in range(0, n+1):
        D[i,0] = i
    for j in range(0, m+1):
        D[0,j] = j
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            if source[i-1] == target[j-1]:
                D[i,j] = D[i-1, j-1]
            else:
                D[i,j] = min(
                    D[i-1, j] + 1,
                    D[i, j-1] + 1,
                    D[i-1, j-1] + 2
                )

    return D[n-1, m-1]
```

### Accepted 4
```python
def solution(A):
    total = sum(A)
    m = float('inf')
    left_sum = 0

    for n in A[:-1]:
        left_sum += n
        v = abs(total - 2*left_sum)
        if v < m:
            m = v
    return m
```

### Accepted 5
```python
def unlock(m):
    return m.lower().translate(
                       str.maketrans(
                          'abcdefghijklmnopqrstuvwxyz',
                          '22233344455566677778889999'
                                    )
                              )
```

### Accepted 6
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provide function to calculate SDE distance

@auth: <NAME>
@date: 2021/05/05
"""

def SDE(front, values1, values2):
    shifted_dict = {}
    for i in front:
        shifted_dict[i] = [(values1[i], values2[i])]
        shifted_list = []
        for j in front:
            if i == j:
                continue
            else:
                shifted_list.append((min(values1[i], values1[j]), min(values2[i], values2[j])))
        shifted_dict[i].append(shifted_list)
    return shifted_dict
```

### Accepted 7
```python
def pickingNumbers(a):
    # Write your code here
    max = 0
    for i in a:
        c = a.count(i)
        d = a.count(i-1)
        e = c+d
        if e>max:
            max = e
    return max
```

### Accepted 8
```python
# content of test_sample.py
def inc(x: int) -> int:
    return x + 1
```

### Accepted 9
```python
"""
Next lexicographical permutation algorithm

https://www.nayuki.io/page/next-lexicographical-permutation-algorithm
"""


def next_lexo(S):
    b = S[-1]
    for i, a in enumerate(reversed(S[:-1]), 2):
        if a < b:
            # we have the pivot a
            for j, b in enumerate(reversed(S), 1):
                if b > a:
                    F = list(S)
                    F[-i], F[-j] = F[-j], F[-i]
                    F = F[: -i + 1] + sorted(F[-i + 1 :])
                    return "".join(F)
        else:
            b = a
    return "no answer"
```

### Accepted 10
```python
def create_array(n):
    res=[]
    i=1
    while i<=n:
        res.append(i)
        i += 1
    return res
```

### Accepted 11
```python
#!/usr/bin/env python3

def date_time(time):
    months = ["January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"]
    hour, minute = int(time[11:13]), int(time[14:16])
    return f"{int(time[0:2])} {months[int(time[3:5])-1]} {time[6:10]} year {hour} hour{'s' if hour!=1 else ''} {minute} minute{'s' if minute!=1 else ''}"

if __name__ == '__main__':
    print(date_time("01.01.2018 00:00"))
    assert date_time("01.01.2018 00:00") == "1 January 2018 year 0 hours 0 minutes"
    assert date_time("04.08.1984 08:15") == "4 August 1984 year 8 hours 15 minutes"
    assert date_time("17.12.1990 07:42") == "17 December 1990 year 7 hours 42 minutes"
```

### Accepted 12
```python
#Return the count of int(s) in passed array. 
def number_of_occurrences(s, xs):
    return xs.count(s)
```

### Accepted 13
```python
def get_season_things_price(thing, amount, price):

    if thing == 'wheel':
        wheel_price = price[thing]['month'] * amount
        return f'Стоимость составит {wheel_price}/месяц'

    else:
        other_thing_price_week = price[thing]['week'] * amount
        other_thing_price_month = price[thing]['month'] * amount
        return f'Стоимость составит {other_thing_price_week} р./неделю' + \
            f' или {other_thing_price_month} р./месяц'
```

### Accepted 14
```python
def main(request, response):
    headers = [("Content-type", "text/html;charset=shift-jis")]
    # Shift-JIS bytes for katakana TE SU TO ('test')
    content =  chr(0x83) + chr(0x65) + chr(0x83) + chr(0x58) + chr(0x83) + chr(0x67);

    return headers, content
```

### Accepted 15
```python
#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#Author:贾江超

def spin_words(sentence):
    list1=sentence.split()
    l=len(list1)
    for i in range(l):
        relen = len(sentence.split()[i:][0])
        if relen > 5:
           list1[i]=list1[i][::-1]
    return ' '.join(list1)

'''
注意 在2.x版本可以用len()得到list的长度 3.x版本就不行了

优化版本  

def spin_words(sentence):
    # Your code goes here
    return " ".join([x[::-1] if len(x) >= 5 else x for x in sentence.split(" ")])
    
    在这里倒序字符串用切片很方便 str[::-1] 就ok了
'''
```

### Accepted 16
```python
def selection_sort(A): # O(n^2)
    n = len(A)
    for i in range(n-1): # percorre a lista
        min = i
        for j in range(i+1, n): # encontra o menor elemento da lista a partir de i + 1
            if A[j] < A[min]:
                min = j
        A[i], A[min] = A[min], A[i] # insere o elemento na posicao correta
    return A

# 1 + (n-1)*[3 + X] = 1 + 3*(n-1) + X*(n-1) = 1 + 3*(n-1) + (n^2 + n - 2)/2
# = (1 - 3 - 1) + (3n + n/2) + (n^2/2)
# The complexity is O(n^2)
```

### Accepted 17
```python
'''
Problem description:
Given a string, determine whether or not the parentheses are balanced
'''


def balanced_parens(str):
    '''
    runtime: O(n)
    space  : O(1)
    '''
    if str is None:
        return True

    open_count = 0

    for char in str:
        if char == '(':
            open_count += 1
        elif char == ')':
            open_count -= 1
            if open_count < 0:
                return False

    return open_count == 0
```

### Accepted 18
```python
def getRoot(config):
  if not config.parent:
    return config
  return getRoot(config.parent)

root = getRoot(config)

# We only run a small set of tests on Windows for now.
# Override the parent directory's "unsupported" decision until we can handle
# all of its tests.
if root.host_os in ['Windows']:
  config.unsupported = False
else:
  config.unsupported = True
```

### Accepted 19
```python
def multiple_replace(text: str, chars_to_mapping: dict):
    """
    This function is used to replace a dictionary of characters inside a text string
    :param text:
    :param chars_to_mapping:
    :return:
    """
    import re

    pattern = "|".join(map(re.escape, chars_to_mapping.keys()))
    return re.sub(pattern, lambda m: chars_to_mapping[m.group()], str(text))
```

### Accepted 20
```python
#!/usr/bin/env python
"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Ambari Agent

"""
import re
__all__ = ["get_bare_principal"]


def get_bare_principal(normalized_principal_name):
  """
  Given a normalized principal name (nimbus/c6501.ambari.apache.org@EXAMPLE.COM) returns just the
  primary component (nimbus)
  :param normalized_principal_name: a string containing the principal name to process
  :return: a string containing the primary component value or None if not valid
  """

  bare_principal = None

  if normalized_principal_name:
    match = re.match(r"([^/@]+)(?:/[^@])?(?:@.*)?", normalized_principal_name)

    if match:
      bare_principal = match.group(1)

  return bare_principal
```

### Accepted 21
```python
def is_leap(year):
    leap=False
    if year%400==0:
        leap=True
    elif year%4==0 and year%100!=0:
        leap=True
    else:
        leap=False
    return leap

year = int(input())
```

### Accepted 22
```python
n = int(input())
row = 0
for i in range(100):
    if 2 ** i <= n <= 2 ** (i + 1) - 1:
        row = i
        break


def seki(k, n):
    for _ in range(n):
        k = 4 * k + 2
    return k


k = 0
if row % 2 != 0:
    k = 2
    cri = seki(k, row // 2)
    if n < cri:
        print("Aoki")
    else:
        print("Takahashi")
else:
    k = 1
    cri = seki(k, row // 2)
    if n < cri:
        print("Takahashi")
    else:
        print("Aoki")
```

### Accepted 23
```python
ans = dict()
pairs = dict()
def create_tree(p):
    if p in ans:
        return ans[p]
    else:
        try:
            res = 0
            if p in pairs:
                for ch in pairs[p]:
                    res += create_tree(ch) + 1
            ans[p] = res
            return res
        except:
            pass
n = int(input())
for i in range(0, n-1):
    child, parent = input().split()
    if parent in pairs:
        pairs[parent].append(child)
    else:
        pairs[parent] = [child]
if n > 0:
    for k in pairs:
        create_tree(k)
    for key in sorted(ans.keys()):
        print(key, ans[key])
```

### Accepted 24
```python
def sysrc(value):
    """Call sysrc.
    CLI Example:

    .. code-block:: bash

        salt '*' freebsd_common.sysrc sshd_enable=YES
        salt '*' freebsd_common.sysrc static_routes
    """
    return __salt__['cmd.run_all']("sysrc %s" % value)
```

### Accepted 25
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:02:33 2019

@author: sercangul
"""

def maxConsecutiveOnes(x): 
  
    # Initialize result 
    count = 0
   
    # Count the number of iterations to 
    # reach x = 0. 
    while (x!=0): 
      
        # This operation reduces length 
        # of every sequence of 1s by one. 
        x = (x & (x << 1)) 
   
        count=count+1
      
    return count

if __name__ == '__main__':
    n = int(input())
    result = maxConsecutiveOnes(n)
    print(result)
```

### Accepted 26
```python
import re


def remove_not_alpha_num(string):
    return re.sub('[^0-9a-zA-Z]+', '', string)


if __name__ == '__main__':
    print(remove_not_alpha_num('a000 aa-b') == 'a000aab')
```

### Accepted 27
```python
##############################################
# The MIT License (MIT)
# Copyright (c) 2014 <NAME>
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
from math import atan, pi


def fov(w,f):
        """
        Returns the FOV as in degrees, given:
            w: image width (or height) in pixels
            f: focalLength (fx or fy) in pixels
        """
        return 2*atan(w/2/f) * 180/pi
```

### Accepted 28
```python
def elo(winner_rank, loser_rank, weighting):
    """
    :param winner: The Player that won the match.
    :param loser: The Player that lost the match.
    :param weighting: The weighting factor to suit your comp.
    :return: (winner_new_rank, loser_new_rank) Tuple.

    This follows the ELO ranking method.
    """
    winner_rank_transformed = 10 ** (winner_rank / 400)
    opponent_rank_transformed = 10 ** (loser_rank / 400)
    transformed_sum = winner_rank_transformed + opponent_rank_transformed

    winner_score = winner_rank_transformed / transformed_sum
    loser_score = opponent_rank_transformed / transformed_sum

    winner_rank = winner_rank + weighting * (
        1 - winner_score)
    loser_rank = loser_rank - weighting * loser_score

    # Set a floor of 100 for the rankings.
    winner_rank = 100 if winner_rank < 100 else winner_rank
    loser_rank = 100 if loser_rank < 100 else loser_rank

    winner_rank = float('{result:.2f}'.format(result=winner_rank))
    loser_rank = float('{result:.2f}'.format(result=loser_rank))

    return winner_rank, loser_rank
```

### Accepted 29
```python
from collections import deque

def solution(N, bus_stop):
    answer = [[1300 for _ in range(N)] for _ in range(N)]
    bus_stop = [(x-1, y-1) for x,y in bus_stop]
    q = deque(bus_stop)
    for x,y in bus_stop:
        answer[x][y] = 0

    while q:
        x, y = q.popleft()
        for nx, ny in ((x-1, y), (x+1, y), (x, y+1), (x, y-1)):
            if (
                0 <= nx < N and 0 <= ny < N
                and answer[nx][ny] > answer[x][y]
            ):
                answer[nx][ny] = answer[x][y] + 1
                q.append((nx, ny))
    return answer

if __name__ == '__main__':
    print(solution(
        3, [[1,2],[3,3]],
    ))
```

### Accepted 30
```python
def insert_metatable():
    """SQL query to insert records from table insert into a table on a DB
    """

    return """
    INSERT INTO TABLE {{ params.target_schema }}.{{ params.target_table }} VALUES 
    ('{{ params.schema }}', '{{ params.table }}', {{ ti.xcom_pull(key='hive_res', task_ids=params.count_inserts)[0][0] }}, current_timestamp(), '{{ params.type }}');
    """
```

### Accepted 31
```python
def save_form(form, actor=None):
    """Allows storing a form with a passed actor. Normally, Form.save() does not accept an actor, but if you require
    this to be passed (is not handled by middleware), you can use this to replace form.save().

    Requires you to use the audit.Model model as the actor is passed to the object's save method.
    """

    obj = form.save(commit=False)
    obj.save(actor=actor)
    form.save_m2m()
    return obj

#def intermediate_save(instance, actor=None):
#    """Allows saving of an instance, without storing the changes, but keeping the history. This allows you to perform
#    intermediate saves:
#
#    obj.value1 = 1
#    intermediate_save(obj)
#    obj.value2 = 2
#    obj.save()
#    <value 1 and value 2 are both stored in the database>
#    """
#    if hasattr(instance, '_audit_changes'):
#        tmp = instance._audit_changes
#        if actor:
#            instance.save(actor=actor)
#        else:
#            instance.save()
#        instance._audit_changes = tmp
#    else:
#        if actor:
#            instance.save(actor=actor)
#        else:
#            instance.save()
```

### Accepted 32
```python
MUTATION = '''mutation {{
  {mutation}
}}'''


def _verify_additional_type(additionaltype):
    """Check that the input to additionaltype is a list of strings.
    If it is empty, raise ValueError
    If it is a string, convert it to a list of strings."""
    if additionaltype is None:
        return None

    if isinstance(additionaltype, str):
        additionaltype = [additionaltype]
    if len(additionaltype) == 0:
        raise ValueError("additionaltype must be a non-empty list")
    return additionaltype
```

### Accepted 33
```python
# written by abraham on aug 24


def dyear2date(dyear):

	year = int(dyear)

	month_lengths = [31,28,31,30,31,30,31,31,30,31,30,31]
	days_before_months = [0,31,59,90,120,151,181,212,243,273,304,334]

	days_into_year_f = (dyear-year)*365
	days_into_year_i = int(days_into_year_f)

	for i in range(12):
		if days_before_months[i] < days_into_year_f < (days_before_months[i]+month_lengths[i]):
			month = i+1
			break

	date = days_into_year_i - days_before_months[month-1]
	hours_f = (days_into_year_f-days_into_year_i)*24
	hours_i = int(hours_f)
	minutes_f = (hours_f-hours_i)*60
	minutes_i = int(minutes_f)
	seconds_i = int((minutes_f-minutes_i)*60)

	return "%02d/%02d/%d %02d:%02d:%02d" % (month,date,year,hours_i,minutes_i,seconds_i)
```

### Accepted 34
```python
def integrate_exponential(a, x0, dt, T):
  """Compute solution of the differential equation xdot=a*x with 
  initial condition x0 for a duration T. Use time step dt for numerical
  solution.
  
  Args:
    a (scalar): parameter of xdot (xdot=a*x)
    x0 (scalar): initial condition (x at time 0)
    dt (scalar): timestep of the simulation
    T (scalar): total duration of the simulation

  Returns:
    ndarray, ndarray: `x` for all simulation steps and the time `t` at each step
  """

  # Initialize variables
  t = np.arange(0, T, dt)
  x = np.zeros_like(t, dtype=complex)
  x[0] = x0

  # Step through system and integrate in time
  for k in range(1, len(t)):
    # for each point in time, compute xdot = a*x
    xdot = (a*x[k-1])

    # update x by adding xdot scaled by dt
    x[k] = x[k-1] +  xdot * dt

  return x, t

# choose parameters
a = -0.5    # parameter in f(x)
T = 10      # total Time duration
dt = 0.001  # timestep of our simulation
x0 = 1.     # initial condition of x at time 0

x, t = integrate_exponential(a, x0, dt, T)
with plt.xkcd():
  fig = plt.figure(figsize=(8, 6))
  plt.plot(t, x.real)  
  plt.xlabel('Time (s)')
  plt.ylabel('x')
```

### Accepted 35
```python
def decimal_to_binary_fraction(x=0.5):
    """
    Input: x, a float between 0 and 1
    Returns binary representation of x
    """
    p = 0
    while ((2 ** p) * x) % 1 != 0:
        # print('Remainder = ' + str((2**p)*x - int((2**p)*x)))
        p += 1

    num = int(x * (2 ** p))

    result = ''
    if num == 0:
        result = '0'
    while num > 0:
        result = str(num % 2) + result
        num //= 2

    for i in range(p - len(result)):
        result = '0' + result

    result = result[0:-p] + '.' + result[-p:]

    return result  # If there is no integer p such that x*(2**p) is a whole number, then internal
# representation is always an approximation

# Suggest that testing equality of floats is not exact: Use abs(x-y) < some
#   small number, rather than x == y

# Why does print(0.1) return 0.1, if not exact?
# Because Python designers set it up this way to automatically round
```

### Accepted 36
```python
def count_words(sentence):
    
    sentence = sentence.lower()
    words = {}
    shit = ',\n:!&@$%^&._'

    for s in shit:
        sentence = sentence.replace(s, ' ')

    for w in sentence.split():
        if w.endswith('\''):
            w = w[:-1]
        if w.startswith('\''):
            w = w[1:]
        words[w] = words.get(w, 0) + 1
    return words
```

### Accepted 37
```python
def load(h):
    return ({'abbr': 0, 'code': 0, 'title': 'There is no appended list'},
            {'abbr': 1,
             'code': 1,
             'title': 'Numbers define number of points corresponding to full coordinate '
                      'circles (i.e. parallels), coordinate values on each circle are '
                      'multiple of the circle mesh, and extreme coordinate values given '
                      'in grid definition (i.e. extreme longitudes) may not be reached in '
                      'all rows'},
            {'abbr': 2,
             'code': 2,
             'title': 'Numbers define number of points corresponding to coordinate lines '
                      'delimited by extreme coordinate values given in grid definition '
                      '(i.e. extreme longitudes) which are present in each row'},
            {'abbr': 3,
             'code': 3,
             'title': 'Numbers define the actual latitudes for each row in the grid. The '
                      'list of numbers are integer values of the valid latitudes in '
                      'microdegrees (scaled by 10-6) or in unit equal to the ratio of the '
                      'basic angle and the subdivisions number for each row, in the same '
                      'order as specified in the scanning mode flag',
             'units': 'bit no. 2'},
            {'abbr': None, 'code': 255, 'title': 'Missing'})
```

### Accepted 38
```python
import re
from collections import Counter

def is_isogram(word):
    if not isinstance(word, str) or word == '': return False
    word = {j for i,j in Counter(
                                 re.sub('[^a-z]', '', word.lower())
                                 ).most_common()
                                                }
    return len(word) == 1
```

### Accepted 39
```python
def removeLoop(head):
    ptr = head
    ptr2 = head
    
    while True :
        if ptr is None or ptr2 is None or ptr2.next is None :
            return
        ptr = ptr.next
        ptr2 = ptr2.next.next
        if ptr is ptr2 :
            loopNode = ptr
            break
    
    ptr = loopNode.next
    count = 1
    while ptr is not loopNode :
        ptr = ptr.next
        count += 1
    
    
    ptr = head
    ptr1 = head
    ptr2 = head.next
    while count > 1 :
        ptr2 = ptr2.next
        ptr1 = ptr1.next
        count -= 1
    
    
    while ptr is not ptr2 :
        ptr = ptr.next
        ptr2 = ptr2.next
        ptr1 = ptr1.next
    
    ptr1.next = None
```

### Accepted 40
```python
def dbl():
    return (
        (a, a) for a in [])
```

### Accepted 41
```python
indexWords = list()

def PreviousWord(_list, _word):
    if _list[_list.index(_word)-1] :
        return _list[_list.index(_word)-1]
    else:
        return
        
phrase = str(input())
phraseList = phrase.split(" ")
length = len(phraseList)
for item in phraseList :
    item = item.strip()

if phrase != "" :
    for i in range(1, length-1) :
        lengthOfWord = len(phraseList[i])
        if phraseList[i][0].isupper() :
            if PreviousWord(phraseList, phraseList[i])[-1] != "." :
                if phraseList[i][-1]=="." or phraseList[i][-1]=="," :
                    indexWords.append(i + 1)
                    indexWords.append(phraseList[i][: lengthOfWord-1]) 
                elif phraseList[i][-1]== "]" and phraseList[i][-2]== "'" :
                    indexWords.append(i + 1)
                    indexWords.append(phraseList[i][: lengthOfWord-2])  
                else :
                    indexWords.append(i + 1)
                    indexWords.append(phraseList[i])
else:
    print("None")

lengthOfIndexWord = len(indexWords)

if lengthOfIndexWord == 0 :
    print("None")
else:
    for i in range(0, lengthOfIndexWord//2):
        print("%i:%s" %(indexWords[2*i],indexWords[(2*i)+1]))
```

### Accepted 42
```python
def prepare_sets(dataset, feature_columns, y_column):
    train_X, val_X, train_y, val_y = train_test_split(dataset[feature_columns], dataset[y_column], random_state=1)
    return train_X, val_X, train_y, val_y
```

### Accepted 43
```python
CHARACTERS_PER_LINE = 39


def break_lines(text):
    chars_in_line = 1
    final_text = ''
    skip = False
    for char in text:
        if chars_in_line >= CHARACTERS_PER_LINE:
            if char == ' ':
                # we happen to be on a space, se we can just break here
                final_text += '\n'
                skip = True
            else:
                # work backwards to find the space to break on
                for i in range(len(final_text) - 1, 0, -1):
                    if final_text[i] == ' ':
                        final_text = final_text[:i] + '\n' + final_text[i + 1:]
                        break
            chars_in_line = 0
        chars_in_line += 1
        if not skip:
            final_text += char
        skip = False
    return final_text


if __name__ == '__main__':
    print(break_lines('The <y<Spirit of the Sword>> guides the goddess\' chosen hero to <r<Skyloft Village>>'))
    print(break_lines('Hey, you look like you have a Questions?'))
    print(break_lines('Skyloft Peater/Peatrice\'s Crystals has Bug Net'))
```

### Accepted 44
```python
from typing import Callable, Iterable, TypeVar

T = TypeVar('T')
Num = TypeVar('Num', int, float)


def sumBy(array: Iterable[T], iteratee: Callable[[T], Num] = None, start: Num = 0) -> Num:
    if iteratee is None:
        return sum([y for y in array], start)
    return sum([iteratee(y) for y in array], start)
```

### Accepted 45
```python
#-- THIS LINE SHOULD BE THE FIRST LINE OF YOUR SUBMISSION! --#

def tally(costs, discounts, rebate_factor):
    cost = sum(costs)
    discount = sum(discounts)

    pre = (cost - discount) * rebate_factor

    if pre < 0:
        return 0
    else:
        return round(pre, 2)

#-- THIS LINE SHOULD BE THE LAST LINE OF YOUR SUBMISSION! ---#

### DO NOT SUBMIT THE FOLLOWING LINES!!! THESE ARE FOR LOCAL TESTING ONLY!
# ((10+24) - (3+4+3)) * 0.3
assert(tally([10,24], [3,4,3], 0.30) == 7.20)
# if the result would be negative, 0 is returned instead
assert(tally([10], [20], 0.1) == 0)
```

### Accepted 46
```python
def apply(con, target_language="E"):
    dict_field_desc = {}
    try:
        df = con.prepare_and_execute_query("DD03T", ["DDLANGUAGE", "FIELDNAME", "DDTEXT"], " WHERE DDLANGUAGE = '"+target_language+"'")
        stream = df.to_dict("records")
        for el in stream:
            dict_field_desc[el["FIELDNAME"]] = el["DDTEXT"]
    except:
        pass
    return dict_field_desc
```

### Accepted 47
```python
def binarySearch(inputArray, searchElement):
    
    minIndex = -1
    maxIndex = len(inputArray)

    while minIndex < maxIndex - 1:
        currentIndex = (minIndex + maxIndex) // 2
        currentElement = inputArray[currentIndex]

        if currentElement < searchElement:
            minIndex = currentIndex
        else:
            maxIndex = currentIndex

    if maxIndex == len(inputArray) or inputArray[maxIndex] != searchElement:
        return -1
    return maxIndex
```

### Accepted 48
```python
def remove_duplicates(lst):
  new = []
  for x in lst:
    if x not in new:
      new.append(x)
  return new
```

### Accepted 49
```python
def three_sum(nums):
    """
    Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0?
    Find all unique triplets in the array which gives the sum of zero.
    :param nums: list[int]
    :return: list[list[int]]
    """
    if len(nums) < 3:
        return []

    nums.sort()
    res = []

    for i in range(len(nums) - 2):
        if i > 0 and nums[i - 1] == nums[i]: continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == 0:
                res.append([nums[i], nums[l], nums[r]])
                l += 1;
                r -= 1
                while l < r and nums[l] == nums[l - 1]: l += 1
                while l < r and nums[r] == nums[r + 1]: r -= 1
            elif s < 0:
                l += 1
            else:
                r -= 1
    return res
```

### Accepted 50
```python
#Exercise: Try to make a function that accepts a function of only positional arguments and returns a function that takes the same number of positional arguments and, given they are all iterators, attempts every combination of one arguments from each iterator.
#Skills: Partial application, Iteration

papplycomboreverse = lambda fun, xiter : lambda *args : [fun(*args, x) for x in xiter]

def combo(fun):
    def returnfun(*args):
        currfun = fun
        for arg in reversed(args):
            currfun = papplycomboreverse(currfun, arg)
        return currfun()
    return returnfun
```

## 50 Rejected Examples with Reasons

### Rejected 1
**Reason:** Too long (>1500 chars)
```python
<reponame>MTES-MCT/sparte
from rest_framework_gis import serializers
from rest_framework import serializers as s

from .models import (
    Artificialisee2015to2018,
    Artificielle2018,
    CommunesSybarval,
    CouvertureSol,
    EnveloppeUrbaine2018,
    Ocsge,
    Renaturee2018to2015,
    Sybar...
```

### Rejected 2
**Reason:** Contains banned keyword: django
```python
from django.contrib import admin
from .models import SearchResult

# Register your models here.
class SearchResultAdmin(admin.ModelAdmin):
    fields = ["query", "heading", "url", "text"]

admin.site.register(SearchResult, SearchResultAdmin)...
```

### Rejected 3
**Reason:** Too long (>1500 chars)
```python
import asyncio
import os
import tempfile
from contextlib import ExitStack
from typing import Text, Optional, List, Union, Dict

from rasa.importers.importer import TrainingDataImporter
from rasa import model
from rasa.model import FingerprintComparisonResult
from rasa.core.domain import Domain
from ...
```

### Rejected 4
**Reason:** SyntaxError/ParseFail: SyntaxError
```python
<gh_stars>1-10
class Solution:
    def finalPrices(self, prices: List[int]) -> List[int]:
        res = []
        for i in range(len(prices)):
            for j in range(i+1,len(prices)):
                if prices[j]<=prices[i]:
                    res.append(prices[i]-prices[j])
                  ...
```

### Rejected 5
**Reason:** Too long (>1500 chars)
```python
<gh_stars>0
# ============================================================================
# FILE: default.py
# AUTHOR: <NAME> <<EMAIL> at g<EMAIL>>
# License: MIT license
# ============================================================================

import re
import typing

from denite.util import...
```

### Rejected 6
**Reason:** Contains banned keyword: open(
```python
<filename>PyDSTool/core/context_managers.py
# -*- coding: utf-8 -*-

"""Context managers implemented for (mostly) internal use"""

import contextlib
import functools
from io import UnsupportedOperation
import os
import sys


__all__ = ["RedirectStdout", "RedirectStderr"]


@contextlib.contextmanager...
```

### Rejected 7
**Reason:** Too long (>1500 chars)
```python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from . import __version__ as app_version

app_name = "pos_kiosk"
app_title = "Pos Kiosk"
app_publisher = "9t9it"
app_description = "Kiosk App"
app_icon = "octicon octicon-file-directory"
app_color = "grey"
app_email = "<EMAIL>"
app_lice...
```

### Rejected 8
**Reason:** Too long (>1500 chars)
```python
<gh_stars>1-10
from keras import Model, Input
from keras.layers import Dense, concatenate, LSTM, Reshape, Permute, Embedding, Dropout, Convolution1D, Flatten
from keras.optimizers import Adam

from pypagai.models.base import KerasModel


class SimpleLSTM(KerasModel):
    """
    Use a simple lstm ne...
```

### Rejected 9
**Reason:** SyntaxError/ParseFail: SyntaxError
```python
<filename>lib/variables/latent_variables/__init__.py
from .fully_connected import FullyConnectedLatentVariable
from .convolutional import ConvolutionalLatentVariable...
```

### Rejected 10
**Reason:** Too long (>1500 chars)
```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
''' PNASNet in PyTorch.
Paper: Progressive Neural Architecture Search
'''

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_...
```

### Rejected 11
**Reason:** Too long (>1500 chars)
```python
# -*- coding: utf-8 -*-
#  coding=utf-8
import json
import os
import math
import logging
import requests
import time

from map_download.cmd.BaseDownloader import DownloadEngine, BaseDownloaderThread, latlng2tile_terrain, BoundBox


def get_access_token(token):
    resp = None
    request_count = 0
 ...
```

### Rejected 12
**Reason:** Too long (>1500 chars)
```python
<reponame>vahini01/electoral_rolls
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 23:28:58 2017

@author: dhingratul
"""
import urllib.request
import os
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup
import ssl
...
```

### Rejected 13
**Reason:** Too long (>1500 chars)
```python
<gh_stars>0
"""
Experiment summary
------------------
Treat each province/state in a country cases over time
as a vector, do a simple K-Nearest Neighbor between
countries. What country has the most similar trajectory
to a given country?

Plots similar countries
"""

import sys
sys.path.insert(0, '.....
```

### Rejected 14
**Reason:** Too long (>1500 chars)
```python
<reponame>steven-lang/rational_activations
"""
Rational Activation Functions for MXNET
=======================================

This module allows you to create Rational Neural Networks using Learnable
Rational activation functions with MXNET networks.
"""
import mxnet as mx
from mxnet import initia...
```

### Rejected 15
**Reason:** Contains banned keyword: torch
```python
<filename>torchflare/criterion/utils.py<gh_stars>1-10
"""Utils for criterion."""
import torch
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Performs L2-Norm."""
    num = x
    denom = torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12
    return num / denom


# Source :...
```

### Rejected 16
**Reason:** Contains 0 functions (need exactly 1)
```python
"""Tests for the sbahn_munich integration"""


line_dict = {
    "name": "S3",
    "color": "#333333",
    "text_color": "#444444",
}...
```

### Rejected 17
**Reason:** Contains banned keyword: flask
```python
<reponame>geudrik/hautomation
#! /usr/bin/env python2.7
# -*- coding: latin-1 -*-

from flask import Blueprint
from flask import current_app
from flask import render_template

from flask_login import login_required

homestack = Blueprint("homestack", __name__, url_prefix="/homestack")


@homestack.r...
```

### Rejected 18
**Reason:** Too long (>1500 chars)
```python
"""Forms for RTD donations"""

import logging

from django import forms
from django.conf import settings
from django.utils.translation import ugettext_lazy as _

from readthedocs.payments.forms import StripeModelForm, StripeResourceMixin
from readthedocs.payments.utils import stripe

from .models im...
```

### Rejected 19
**Reason:** Too long (>1500 chars)
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base import DataReaderBase
from ..tools import COL, _get_dates, to_float, to_int

import pandas as pd
#from pandas.tseries.frequencies import to_offset
from six.moves import cStringIO as StringIO
import logging
import traceback
import datetime

im...
```

### Rejected 20
**Reason:** Too long (>1500 chars)
```python
<reponame>Vail-qin/Keras-TextClassification
# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/2 21:08
# @author  : Mo
# @function:


from keras_textclassification.data_preprocess.text_preprocess import load_json, save_json
from keras_textclassification.conf.path_config import path_mode...
```

### Rejected 21
**Reason:** Too long (>1500 chars)
```python
# Copyright 2014 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

from gpu_tests.gpu_test_expectations import GpuTestExpectations

# See the GpuTestExpectations class for documentation.

class PixelExpec...
```

### Rejected 22
**Reason:** Too long (>1500 chars)
```python
<filename>examples/p02_budgets/budget_data_ingest/migrations/0001_initial.py
# -*- coding: utf-8 -*-
# Generated by Django 1.11.13 on 2018-06-08 22:54
from __future__ import unicode_literals

from django.conf import settings
import django.contrib.postgres.fields.jsonb
from django.db import migration...
```

### Rejected 23
**Reason:** Contains banned keyword: open(
```python
import setuptools  #enables develop

setuptools.setup(
    name='pysvm',
    version='0.1',
    description='PySVM : A NumPy implementation of SVM based on SMO algorithm',
    author_email="<EMAIL>",
    packages=['pysvm'],
    license='MIT License',
    long_description=open('README.md', encoding='...
```

### Rejected 24
**Reason:** Too long (>1500 chars)
```python
<gh_stars>1-10
######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: <NAME>
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
...
```

### Rejected 25
**Reason:** Disallowed importFrom: data_collection
```python
from data_collection.management.commands import BaseXpressDemocracyClubCsvImporter

class Command(BaseXpressDemocracyClubCsvImporter):
    council_id = 'E06000027'
    addresses_name = 'parl.2017-06-08/Version 1/Torbay Democracy_Club__08June2017.tsv'
    stations_name = 'parl.2017-06-08/Version 1/To...
```

### Rejected 26
**Reason:** Too long (>1500 chars)
```python
from django.db.models import Q

from django.shortcuts import render
from django.http import Http404

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view

from .models import Product, Category
f...
```

### Rejected 27
**Reason:** Disallowed importFrom: sys
```python
from sys import maxsize


class Contact:

    def __init__(self, fname=None, mname=None, lname=None, nick=None, title=None, comp=None, addr=None,
                 home=None, mobile=None, work=None, fax=None, email1=None, email2=None, email3=None,
                 homepage=None, bday=None, bmonth=Non...
```

### Rejected 28
**Reason:** Too long (>1500 chars)
```python
##########################################################################
#
#  Copyright (c) 2010-2012, Image Engine Design Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  ...
```

### Rejected 29
**Reason:** Too long (>1500 chars)
```python
<filename>rlpy/Domains/Pacman.py
"""Pacman game domain."""
from rlpy.Tools import __rlpy_location__
from .Domain import Domain
from .PacmanPackage import layout, pacman, game, ghostAgents
from .PacmanPackage import graphicsDisplay
import numpy as np
from copy import deepcopy
import os
import time

_...
```

### Rejected 30
**Reason:** Disallowed importFrom: zeit
```python
from zeit.cms.i18n import MessageFactory as _
import zope.interface
import zope.schema


class IGlobalSettings(zope.interface.Interface):
    """Global CMS settings."""

    default_year = zope.schema.Int(
        title=_("Default year"),
        min=1900,
        max=2100)

    default_volume = zop...
```

### Rejected 31
**Reason:** SyntaxError/ParseFail: SyntaxError
```python
<filename>abc/abc165/abc165e.py
N, M = map(int, input().split())

for i in range(1, M + 1):
    if i % 2 == 1:
        j = (i - 1) // 2
        print(1 + j, M + 1 - j)
    else:
        j = (i - 2) // 2
        print(M + 2 + j, 2 * M + 1 - j)...
```

### Rejected 32
**Reason:** Too long (>1500 chars)
```python
<reponame>giggslam/python-messengerbot-sdk<filename>setup.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Licensed under the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License. You may obtain
#  a copy of the License at
#
#       http:...
```

### Rejected 33
**Reason:** Too long (>1500 chars)
```python
<gh_stars>1-10
# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#...
```

### Rejected 34
**Reason:** Too long (>1500 chars)
```python
<filename>eth2/beacon/chains/base.py
from abc import (
    ABC,
    abstractmethod,
)
import logging
from typing import (
    TYPE_CHECKING,
    Tuple,
    Type,
)

from eth._utils.datatypes import (
    Configurable,
)
from eth.db.backends.base import (
    BaseAtomicDB,
)
from eth.exceptions impor...
```

### Rejected 35
**Reason:** Too long (>1500 chars)
```python
#!/usr/local/bin/python3

import paramiko,time

#using as SSH Client

client = paramiko.SSHClient()

# check dir(client) to find available options.
# auto adjust host key verification with yes or no
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# time for connecting to remote Cisco I...
```

### Rejected 36
**Reason:** Contains 2 functions (need exactly 1)
```python
# for n in range(400,500):
#     i = n // 100
#     j = n // 10 % 10
#     k = n % 10
#     if n == i ** 3 + j ** 3 + k ** 3:
#         print(n)


# 第一道题(16)
# input("请输入(第一次):")
# s1 = input("请输入(第二次):")

# l1 = s1.split(' ')
# l2 = []
# for i in l1:
#     if i.isdigit():
#         l2.append(int(i)...
```

### Rejected 37
**Reason:** Too long (>1500 chars)
```python
<filename>graphdb/transformer.py<gh_stars>1-10
"""
A query transformer is a function that accepts a program and returns a program, plus a priority level.
Higher priority transformers are placed closer to the front of the list. We’re ensuring is a function,
because we’re going to evaluate it later 31...
```

### Rejected 38
**Reason:** SyntaxError/ParseFail: SyntaxError
```python
<filename>yzcore/templates/project_template/src/const/_job.py
#!/usr/bin/python3.6.8+
# -*- coding:utf-8 -*-
"""
@auth: cml
@date: 2020-12-2
@desc: ...
"""


class JobStatus(object):
    PENDING = 0  # 任务等待执行

    STARTED = 100  # 任务执行开始
    PROCESS = 110
    POLLING = 120
    CALLBACK = 130

    SU...
```

### Rejected 39
**Reason:** Too long (>1500 chars)
```python
<gh_stars>0
# -*- coding: utf-8 -*-
"""
    pyboleto.html
    ~~~~~~~~~~~~~

    Classe Responsável por fazer o output do boleto em html.

    :copyright: © 2012 by <NAME>
    :license: BSD, see LICENSE for more details.

"""
import os
import string
import sys
import codecs
import base64

from itert...
```

### Rejected 40
**Reason:** Contains 0 functions (need exactly 1)
```python
summary = 0

i = 0
while i < 5:
    summary = summary + i
    print(summary)
    i = i + 1...
```

### Rejected 41
**Reason:** Disallowed import: imtreat
```python
import imtreat

img = imtreat.imageManagerClass.openImageFunction("../images/soleil.png", 0)

img = imtreat.definedModesClass.detailEnhanceFunction(img)

imtreat.imageManagerClass.saveImageFunction("/Téléchargements/", "image_1", ".png", img)...
```

### Rejected 42
**Reason:** Too long (>1500 chars)
```python
<filename>nova/conf/hyperv.py<gh_stars>0
# Copyright (c) 2016 <NAME>
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www...
```

### Rejected 43
**Reason:** Contains banned keyword: open(
```python
import requests

words_list = requests.get("https://raw.githubusercontent.com/atebits/Words/master/Words/fr.txt").text

words_list = filter(lambda x: len(x) > 4, words_list.split('\n'))

path = input("Chemin d'écriture ? (words.txt) ")

if path == "":
    path = "./words.txt"

with open(path, "w", e...
```

### Rejected 44
**Reason:** Too long (>1500 chars)
```python
import unittest
from unittest import mock
import os
import subprocess
from testfixtures import TempDirectory
from simplegallery.upload.uploader_factory import get_uploader


class AWSUploaderTestCase(unittest.TestCase):

    def test_no_location(self):
        uploader = get_uploader('aws')
        ...
```

### Rejected 45
**Reason:** Too long (>1500 chars)
```python
<reponame>kithsirij/NLP-based-Syllabus-Coverage-Exam-paper-checker-Tool
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'add_subject.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import Qt...
```

### Rejected 46
**Reason:** Too long (>1500 chars)
```python
<gh_stars>1-10
from django.db.models import signals
from django.test import TestCase
from django.core import management
from django.utils import six

from shared_models import models


PRE_SYNCDB_ARGS = ['app', 'create_models', 'verbosity', 'interactive', 'db']
SYNCDB_DATABASE = 'default'
SYNCDB_VER...
```

### Rejected 47
**Reason:** Too long (>1500 chars)
```python
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicabl...
```

### Rejected 48
**Reason:** Contains banned keyword: os.system
```python
<filename>examples/mouse.py
#!/usr/bin/env python
import time
import os
import math
from trackball import TrackBall

print("""Trackball: Mouse

Use the trackball as a mouse in Raspbian, with right-click
when the switch is pressed.

Press Ctrl+C to exit!
""")

trackball = TrackBall(interrupt_pin=4)
t...
```

### Rejected 49
**Reason:** Too long (>1500 chars)
```python
<filename>garaged/src/garage/tf/regressors/gaussian_mlp_regressor_model.py
"""GaussianMLPRegressorModel."""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from garage.experiment import deterministic
from garage.tf.models import GaussianMLPModel


class GaussianMLPRe...
```

### Rejected 50
**Reason:** Too long (>1500 chars)
```python
<filename>test.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import argparse

from ...
```
