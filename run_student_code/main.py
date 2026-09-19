from fastapi import FastAPI
from pydantic import BaseModel
import subprocess

app = FastAPI(title="Python Code Runner API")

class CodeSubmission(BaseModel):
    code: str

def run_student_code(code_string: str) -> str:
    with open("temp_student_code.py", "w") as file:
        file.write(code_string)
    
    try:
        result = subprocess.run(
            ["python", "temp_student_code.py"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode != 0:
            return result.stderr.strip()
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Error: Time Limit Exceeded"

@app.post("/execute")
def execute_code(submission: CodeSubmission):
    output = run_student_code(submission.code)
    return {"output": output}