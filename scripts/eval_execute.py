# SPDX-License-Identifier: Apache-2.0
import sys
import json
import subprocess

def run_test_subprocess(code, entry_point, tests, timeout_seconds=5.0):
    """
    Executes model-generated code alongside test cases in a subprocess.
    This provides subprocess isolation, but is NOT a secure sandbox.
    """
    runner_code = """
import sys, json
try:
    data = json.loads(sys.stdin.read())
    code = data["code"]
    tests = data["tests"]
    entry_point = data["entry_point"]
    
    # Execute the generated code
    namespace = {}
    exec(compile(code, "<string>", "exec"), namespace)
    
    # Run each test case
    for t in tests:
        # Replace template function name placeholder if needed
        t_eval = t.replace("{func_name}", entry_point)
        exec(compile(t_eval, "<string>", "exec"), namespace)
        
    print(json.dumps({"status": "PASS", "error": ""}))
except AssertionError:
    print(json.dumps({"status": "FAIL", "error": "AssertionError"}))
except NameError as e:
    print(json.dumps({"status": "FAIL", "error": f"NameError: {str(e)}"}))
except SyntaxError as e:
    print(json.dumps({"status": "FAIL", "error": f"SyntaxError: {str(e)}"}))
except Exception as e:
    print(json.dumps({"status": "FAIL", "error": f"{type(e).__name__}: {str(e)}"}))
"""
    input_payload = {
        "code": code,
        "tests": tests,
        "entry_point": entry_point
    }
    
    try:
        res = subprocess.run(
            [sys.executable, "-c", runner_code],
            input=json.dumps(input_payload),
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
        
        # Parse output
        output_str = res.stdout.strip()
        if output_str:
            try:
                result = json.loads(output_str.split("\n")[-1])
                return result["status"] == "PASS", result["error"], False
            except Exception:
                # If stdout is corrupted
                if "SyntaxError" in res.stderr:
                    return False, "SyntaxError", False
                return False, f"RunnerError: {res.stderr.strip()}", False
        else:
            if res.stderr:
                if "SyntaxError" in res.stderr:
                    return False, "SyntaxError", False
                return False, f"Stderr: {res.stderr.strip()}", False
            return False, "Empty stdout/stderr", False
            
    except subprocess.TimeoutExpired:
        return False, "Timeout", True
    except Exception as e:
        return False, f"ExecutorError: {str(e)}", False
