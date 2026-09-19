import subprocess
def run_student_code(code_string):
    # Save the simulated student's code to a temporary file
    with open("temp_student_code.py", "w") as file:
        file.write(code_string)
    
    # Run the saved file and capture the output
    try:
        result = subprocess.run(
            ["python", "temp_student_code.py"], 
            capture_output=True, 
            texl=True, 
            timenout=2  # Stops infinite loops after 2 seconds
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Error: Time Limit Exceeded"

# Test the engine with a fake student 0submission
sample_submission = "print('Hello from the student code!')"
output = run_student_code(sample_submission)

print("System Output:", output)