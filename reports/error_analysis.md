# Scientific Error Analysis Report

This report categorizes the failure modes of the curriculum-trained **SamatNext-v0.1** and the matched **Transformer** baseline across the evaluation stages. Errors are categorized using subprocess isolation output.

## SamatNext Curriculum

### STAGE5 (Total Tasks: 500)
- **Pass Rate:** 97.6%
| Failure Mode | Count | Percentage of Failures |
| :--- | :---: | :---: |
| SyntaxError / IndentationError | 0 | 0.0% |
| AssertionError | 0 | 0.0% |
| NameError / AttributeError | 12 | 100.0% |
| Timeout | 0 | 0.0% |
| Other Exceptions | 0 | 0.0% |

### STAGE3 (Total Tasks: 500)
- **Pass Rate:** 86.8%
| Failure Mode | Count | Percentage of Failures |
| :--- | :---: | :---: |
| SyntaxError / IndentationError | 14 | 21.2% |
| AssertionError | 19 | 28.8% |
| NameError / AttributeError | 22 | 33.3% |
| Timeout | 0 | 0.0% |
| Other Exceptions | 11 | 16.7% |

### STAGE2E (Total Tasks: 300)
- **Pass Rate:** 6.3%
| Failure Mode | Count | Percentage of Failures |
| :--- | :---: | :---: |
| SyntaxError / IndentationError | 41 | 14.6% |
| AssertionError | 3 | 1.1% |
| NameError / AttributeError | 234 | 83.3% |
| Timeout | 0 | 0.0% |
| Other Exceptions | 3 | 1.1% |

## Transformer Curriculum

### STAGE5 (Total Tasks: 500)
- **Pass Rate:** 49.4%
| Failure Mode | Count | Percentage of Failures |
| :--- | :---: | :---: |
| SyntaxError / IndentationError | 85 | 33.6% |
| AssertionError | 24 | 9.5% |
| NameError / AttributeError | 125 | 49.4% |
| Timeout | 0 | 0.0% |
| Other Exceptions | 19 | 7.5% |

### STAGE3 (Total Tasks: 500)
- **Pass Rate:** 4.0%
| Failure Mode | Count | Percentage of Failures |
| :--- | :---: | :---: |
| SyntaxError / IndentationError | 61 | 12.7% |
| AssertionError | 2 | 0.4% |
| NameError / AttributeError | 415 | 86.5% |
| Timeout | 0 | 0.0% |
| Other Exceptions | 2 | 0.4% |

### STAGE2E (Total Tasks: 300)
- **Pass Rate:** 0.0%
| Failure Mode | Count | Percentage of Failures |
| :--- | :---: | :---: |
| SyntaxError / IndentationError | 37 | 12.3% |
| AssertionError | 0 | 0.0% |
| NameError / AttributeError | 263 | 87.7% |
| Timeout | 0 | 0.0% |
| Other Exceptions | 0 | 0.0% |

