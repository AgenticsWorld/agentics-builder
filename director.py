import shlex
from pydantic import BaseModel
from typing import Optional, List, Literal
from pathlib import Path
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
import sys
import os
import requests
import yaml
import argparse
import json
import subprocess
import time
import modules.llm as llm


class EvaluationResult(BaseModel):
    success: bool
    feedback: Optional[str]


class DirectorConfig(BaseModel):
    prompt: str
    coder_model: str
    evaluator_model: str
    max_iterations: int
    execution_command: str
    context_editable: List[str]
    context_read_only: List[str]
    evaluator: Literal["default", "none"]
    language: str
    edit_format: str
    editor_model: str
    enable_notes: bool = True

class Director:
    """
    Self Directed AI Coding Assistant
    """

    total_cost = 0.0

    def __init__(self, config_path: str):
        # Load and validate config
        self.config = self.validate_config(Path(config_path))
        # Customize Models based on environment variables
        self.customized_model = os.getenv("CUSTOMIZED_LLM_MODEL")
        # Initialize LLM
        self.llm_client = llm.LLMClient(provider="openrouter", model=self.config.evaluator_model)

    @staticmethod
    def validate_config(config_path: Path) -> DirectorConfig:
        """Validate the yaml config file and return DirectorConfig object."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # If prompt ends with .md, read content from that file
        if config_dict["prompt"].endswith(".md"):
            prompt_path = Path(config_dict["prompt"])
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
            with open(prompt_path) as f:
                config_dict["prompt"] = f.read()

        config = DirectorConfig(**config_dict)

        # Validate we have at least 1 editable file
        if not config.context_editable:
            raise ValueError("At least one editable context file must be specified")

        # Validate all paths in context_editable and context_read_only exist
        for path in config.context_editable:
            if not Path(path).exists():
                raise FileNotFoundError(f"Editable context file not found: {path}")

        for path in config.context_read_only:
            if not Path(path).exists():
                raise FileNotFoundError(f"Read-only context file not found: {path}")

        return config

    def parse_llm_json_response(self, str) -> str:
        """
        Parse and fix the response from an LLM that is expected to return JSON.
        """
        if "```" not in str:
            str = str.strip()
            self.file_log(f"raw pre-json-parse: {str}", print_message=False)
            return str

        # Remove opening backticks and language identifier
        str = str.split("```", 1)[-1].split("\n", 1)[-1]

        # Remove closing backticks
        str = str.rsplit("```", 1)[0]

        str = str.strip()

        self.file_log(f"post-json-parse: {str}", print_message=False)

        # Remove any leading or trailing whitespace
        return str

    def file_log(self, message: str, print_message: bool = True):
        if print_message:
            print(message)
        with open("director_log.txt", "a+") as f:
            f.write(message + "\n")

    # ------------- Key Director Methods -------------

    def create_new_ai_coding_prompt(
        self,
        iteration: int,
        base_input_prompt: str,
        execution_output: str,
        evaluation: EvaluationResult,
    ) -> str:
        if iteration == 0:
            return base_input_prompt
        else:
            return f"""
# Generate the next iteration of code to achieve the user's desired result based on their original instructions and the feedback from the previous attempt.
> Generate a new prompt in the same style as the original instructions for the next iteration of code.

## This is your {iteration}th attempt to generate the code.
> You have {self.config.max_iterations - iteration} attempts remaining.

## Here's the user's original instructions for generating the code:
{base_input_prompt}

## Here's the output of your previous attempt:
{execution_output}

## Here's feedback on your previous attempt:
{evaluation.feedback}"""

    def ai_code(self, prompt: str):
        model = Model(
            self.customized_model or self.config.coder_model,
            editor_model=self.customized_model or self.config.editor_model,
            editor_edit_format="diff",
        )

        coder = Coder.create(
            main_model=model,
            edit_format=self.config.edit_format,
            io=InputOutput(yes=True),
            fnames=self.config.context_editable,
            read_only_fnames=self.config.context_read_only,
            auto_commits=False,
            suggest_shell_commands=False,
            # detect_urls=False,
        )
        coder.run(prompt)

        if coder.total_cost is not None:
            self.total_cost += coder.total_cost

    def ai_note(self, prompt: str, read_only_fnames: List[str]):
        note_model = Model(self.customized_model or "openrouter/deepseek/deepseek-chat-v3-0324")
        noter = Coder.create(
            main_model=note_model,
            io=InputOutput(yes=True),
            fnames=["notes.txt"],
            read_only_fnames=read_only_fnames,
            auto_commits=False,
            suggest_shell_commands=False,
            # detect_urls=False,
        )
        noter.run(prompt)

        if noter.total_cost is not None:
            self.total_cost += noter.total_cost


    def execute(self) -> str:
        """Execute the tests and return the output as a string."""
        r = ""
        if os.getenv("EVALUATE_API"):
            # Get API endpoint from environment variable
            api_url = os.getenv("EVALUATE_API")
            if not api_url:
                self.file_log("EVALUATE_API environment variable is set but empty")
                return "Error: EVALUATE_API environment variable is set but empty"

            headers = {
                "Authorization": f"Bearer {os.getenv('AGENT_ACCESS_TOKEN')}",
            }
            try:
                response = requests.post(api_url, headers=headers)
                response.raise_for_status()
                data = response.json()
                session = data['data']['session']
                if session is None or session == "":
                    raise ValueError("Evaluate request session is None or empty")
                
                # max 2 hours
                max_polls = 3600
                poll_interval = 2
                poll_count = 0

                while poll_count < max_polls:
                    poll_count += 1
                    try:
                        poll_response = requests.get(f"{api_url}/{session}", headers=headers)
                        poll_response.raise_for_status()
                        poll_data = poll_response.json()
                        
                        # Check if the execution is complete
                        if poll_data['data']['status'] > 0:
                            data = poll_data
                            break
                    except Exception as e:
                        self.file_log(f"Evaluate result polling failed: {str(e)}")
                    
                    time.sleep(poll_interval)
                else:
                    raise TimeoutError(f"Execution timed out after {max_polls} polls")
                
                if data['data']['status'] != 1:
                    raise ValueError(f"Execution failed with status: {data['data']['status']}")

                self.file_log(f"Execution output: \n{data['data']['stdout'] + data['data']['stderr']}", print_message=False)
                r = data['data']['stdout'] + data['data']['stderr']

            except Exception as e:
                error_msg = f"API execution failed: {str(e)}"
                self.file_log(error_msg, print_message=False)
                r = error_msg
        else:
            result = subprocess.run(
                shlex.split(self.config.execution_command),
                capture_output=True,
                text=True,
            )
            self.file_log(
                f"Execution output: \n{result.stdout + result.stderr}",
                print_message=False,
            )
            r = result.stdout + result.stderr
        return r

    def evaluate(self, execution_output: str) -> EvaluationResult:

        if self.config.evaluator != "default" and self.config.evaluator != "none":
            raise ValueError(
                f"Custom evaluator {self.config.evaluator} not implemented"
            )

        map_editable_fname_to_files = {
            Path(fname).name: Path(fname).read_text()
            for fname in self.config.context_editable
        }

        map_read_only_fname_to_files = {
            Path(fname).name: Path(fname).read_text()
            for fname in self.config.context_read_only
        }

        evaluation_prompt = f"""Evaluate this execution output and determine if it was successful based on the execution command, the user's desired result, the editable files, checklist, and the read-only files.

## Checklist:
- Is the execution output reporting success or failure?
- Did we miss any tasks? Review the User's Desired Result to see if we have satisfied all tasks.
- Did we satisfy the user's desired result?
- Ignore warnings

## User's Desired Result:
{self.config.prompt}

## Editable Files:
{map_editable_fname_to_files}

## Read-Only Files:
{map_read_only_fname_to_files}

## Execution Command:
{self.config.execution_command}
                                        
## Execution Output:
{execution_output}

## Response Format:
> Be 100% sure to output JSON.parse compatible JSON.
> That means no new lines.

Return a structured JSON response with the following structure: {{
    success: bool - true if the execution output generated by the execution command matches the Users Desired Result
    feedback: str | None - if unsuccessful, provide detailed feedback explaining what failed and how to fix it, or None if successful
}}"""

        self.file_log(
            f"Evaluation prompt: ({self.config.evaluator_model}):\n{evaluation_prompt}",
            print_message=False,
        )

        try:
            completion = self.llm_client.chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": evaluation_prompt,
                    },
                ],
                temperature=0.7,
                max_tokens=50000
            )

            self.file_log(
                f"Evaluation response: ({self.config.evaluator_model}):\n{completion['content']}",
                print_message=False,
            )

            evaluation = EvaluationResult.model_validate_json(
                self.parse_llm_json_response(completion['content'])
            )

            return evaluation
        except Exception as e:

            self.file_log(
                f"Error evaluating execution output for '{self.config.evaluator_model}'. Error: {e}. Falling back to gpt-4o & structured output."
            )

            ## Fallback
            try:
                # Create a new client with the fallback model
                fallback_client = llm.LLMClient(provider="openrouter", model="gpt-4o")
                completion = fallback_client.chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": evaluation_prompt,
                        },
                    ],
                    temperature=0.7,
                    max_tokens=50000
                )
                
                evaluation = EvaluationResult.model_validate_json(
                    self.parse_llm_json_response(completion['content'])
                )
                return evaluation
            except Exception as fallback_error:
                self.file_log(f"Fallback also failed: {fallback_error}")
                raise ValueError("Failed to parse the response")

    def direct(self):
        evaluation = EvaluationResult(success=False, feedback=None)
        execution_output = ""
        success = False

        for i in range(self.config.max_iterations):
            self.file_log(f"\nIteration {i+1}/{self.config.max_iterations}")

            self.file_log("🧠 Creating new prompt...")
            new_prompt = self.create_new_ai_coding_prompt(
                i, self.config.prompt, execution_output, evaluation
            )

            self.file_log("🤖 Generating AI code...")
            self.ai_code(new_prompt)

            # 只有当 self.config.evaluator = "default" 时才执行后续代码
            if self.config.evaluator != "default":
                self.file_log("⏭️ Skipping execution and evaluation steps because evaluator is not 'default'")
                continue

            self.file_log(f"💻 Executing code... '{self.config.execution_command}'")
            execution_output = self.execute()

            self.file_log(
                f"🔍 Evaluating results... '{self.config.evaluator_model}' + '{self.config.evaluator}'"
            )
            evaluation = self.evaluate(execution_output)

            self.file_log(
                f"🔍 Evaluation result: {'✅ Success' if evaluation.success else '❌ Failed'}"
            )
            if evaluation.feedback:
                self.file_log(f"💬 Feedback: \n{evaluation.feedback}")

            if evaluation.success:
                success = True
                self.file_log(
                    f"\n🎉 Success achieved after {i+1} iterations! Breaking out of iteration loop."
                )

                break
            else:
                self.file_log(
                    f"\n🔄 Continuing with next iteration... Have {self.config.max_iterations - i - 1} attempts remaining."
                )

        if not success:
            self.file_log(
                "\n🚫 Failed to achieve success within the maximum number of iterations."
            )

        self.file_log("\nDone.")

        if self.config.enable_notes:
            note_prompt = f"UPDATE `notes.txt`: Assuming agent.py is an AI Agent, please introduce this AI Agent by explaining the functionality of the agent.py code in {self.config.language}, with explanations with explanations that can help non-technical users using `notes.txt` understand and verify the correctness of the program's execution process. Keep the total word count within 200 words, do not include function names, file names, precautions, or program verification methods"
            self.ai_note(note_prompt, ["agent.py"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the AI Coding Director with a config file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="specs/basic.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()
    director = Director(args.config)
    director.direct()
    
    with open('cost.txt', 'w') as f:
        f.write(f"{director.total_cost:.4f}")
