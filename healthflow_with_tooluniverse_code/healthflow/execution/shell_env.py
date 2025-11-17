import os  
import subprocess 
import tempfile 
import traceback

def run_bash_script(script: str, working_dir: str = None) -> str:
    try:
        script = script.strip()

        if not script:
            return "Error: Empty script"

        with tempfile.NamedTemporaryFile(suffix=".sh", mode="w", delete=False) as f:
            if not script.startswith("#!/"):
                f.write("#!/bin/bash\n")

            if "set -e" not in script:
                f.write("set -e\n")
            
            f.write(script)
            temp_file = f.name 

        os.chmod(temp_file, 0o755)

        env = os.environ.copy()
        cwd = working_dir if working_dir else os.getcwd() 

        result = subprocess.run(
            [temp_file],         
            shell=True,          
            capture_output=True,  
            text=True,            
            check=False,          
            env=env,             
            cwd=cwd,             
        )

        os.unlink(temp_file)

        if result.returncode != 0:
            traceback.print_stack()
            print(result)           
            return f"Error running Bash script (exit code {result.returncode}):\n{result.stderr}"
        else:
            return result.stdout
            
    except Exception as e:
        traceback.print_exc()
        return f"Error running Bash script: {str(e)}"