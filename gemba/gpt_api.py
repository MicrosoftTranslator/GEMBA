import sys
import time
import ipdb
import logging
from termcolor import colored
from datetime import datetime
import openai


# class for calling OpenAI API and handling cache
class GptApi:
    def __init__(self, credentials, verbose=True):
        assert "api_key" in credentials, "api_key not found in credentials"
        assert "deployments" in credentials, "deployments not found in credentials"

        self.deployments = credentials["deployments"]
        self.verbose = verbose

        if "api_base" in credentials:
            # Azure API access
            self.api_type = "azure"
            self.client = openai.AzureOpenAI(
                api_key=credentials["api_key"],  
                api_version="2023-12-01-preview",
                azure_endpoint=credentials["api_base"],
                timeout=60
            )
        else:
            # OpenAI API access
            self.api_type = "openai"
            self.client = openai.OpenAI(
                api_key=credentials["api_key"],
                timeout=60
            )

        # limit the number of requests per second
        if "requests_per_second_limit" in credentials and credentials["requests_per_second_limit"] > 0:
            self.rps_limit = 1 / credentials["requests_per_second_limit"]
        else:
            self.rps_limit = 0
        self.last_call_timestamp = 0

        self.non_batchable_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"]
        logging.getLogger().setLevel(logging.CRITICAL) # in order to suppress all these HTTP INFO log messages

    # answer_id is used for determining if it was the top answer or how deep in the list it was
    def request(self, prompt, model, parse_response, temperature=0, answer_id=-1, cache=None, max_tokens=20):
        answers = None
        if cache is not None:
            answers = cache.get({
                "model": model,
                "temperature": temperature,
                "prompt": prompt,
            })

        if answers is None:
            answers = self.request_api(prompt, model, temperature, max_tokens)
            if cache is not None:
                cache.add({
                           "model": model,
                           "temperature": temperature,
                           "prompt": prompt,
                           "answers": answers,
                          })

        # there is no valid answer
        if len(answers) == 0:
            return [{
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": None,
                    "prompt": prompt,
                    "finish_reason": None,
                    "model": model,
                    }]

        parsed_answers = []
        for full_answer in answers:
            finish_reason = full_answer["finish_reason"]
            full_answer = full_answer["answer"]
            answer_id += 1
            answer = parse_response(full_answer)
            if self.verbose or temperature > 0:
                print(f"Answer (t={temperature}): " + colored(answer, "yellow") + " (" + colored(full_answer, "blue") + ")", file=sys.stderr)
            if answer is None:
                continue
            parsed_answers.append(
                {
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": answer,
                    "prompt": prompt,
                    "finish_reason": finish_reason,
                    "model": model,
                }
            )

        # there was no valid answer, increase temperature and try again
        if len(parsed_answers) == 0:
            return self.request(prompt, model, parse_response, temperature=temperature + 1, answer_id=answer_id, cache=cache)

        return parsed_answers

    def request_api(self, prompt, model, temperature=0, max_tokens=20):
        # if temperature is 0, then request only 1 response
        n = 1
        if temperature > 0:
            n = 10
        elif temperature >= 5:
            n = 20

        if max_tokens > 5000 or temperature > 10:
            return []

        dt = datetime.now()
        ts = datetime.timestamp(dt)
        if ts - self.last_call_timestamp < self.rps_limit:
            time.sleep(self.rps_limit - (ts - self.last_call_timestamp))

        self.last_call_timestamp = ts

        while True:
            try:
                response = self.call_api(prompt, model, n, temperature, max_tokens)
                break
            except Exception as e:
                # response was filtered
                if hasattr(e, 'code'):
                    if e.code == 'content_filter':
                        return []
                    print(e.code, file=sys.stderr)
                # frequent error is reaching the API limit
                print(colored("Error, retrying...", "red"), file=sys.stderr)
                print(e, file=sys.stderr)
                time.sleep(1)

        answers = []
        for choice in response.choices:
            if hasattr(choice, "message"):
                answer = choice.message.content.strip()
            else:
                answer = choice.text.strip()
                
            # one of the responses didn't finish, we need to request more tokens
            if choice.finish_reason != "stop":
                if self.verbose:
                    print(colored(f"Increasing max tokens to fit answers.", "red") + colored(answer, "blue"), file=sys.stderr)
                return self.request_api(prompt, model, temperature=temperature, max_tokens=max_tokens + 200)

            answers.append({
                "answer": answer,
                "finish_reason": choice.finish_reason,
            })

        if len(answers) > 1:
            # remove duplicate answers
            answers = [dict(t) for t in {tuple(d.items()) for d in answers}]

        return answers

    def call_api(self, prompt, model, n, temperature, max_tokens):        
        parameters = {
            "temperature": temperature/10,
            "max_tokens": max_tokens,
            "top_p": 1,
            "n": n,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "model": self.deployments[model]
        }

        if model in self.non_batchable_models:
            if isinstance(prompt, list):
                # check that prompt contain list of dictionaries with role and content
                assert all(isinstance(p, dict) for p in prompt), "Prompts must be a list of dictionaries."
                assert all("role" in p and "content" in p for p in prompt), "Prompts must be a list of dictionaries with role and content."

                parameters["messages"] = prompt
            else:
                parameters["messages"] = [{
                    "role": "user",
                    "content": prompt,
                }]

            completion_function = self.client.chat.completions.create
        else:
            # check that prompt is a list of strings
            assert isinstance(prompt, str), "prompt must be a strings."

            parameters["prompt"] = prompt
            completion_function = self.client.completions.create

        return completion_function(**parameters)
    
    def bulk_request(self, df, model, parse_mqm_answer, cache, max_tokens=20):
        answers = []
        for i, row in df.iterrows():
            prompt = row["prompt"]
            parsed_answers = self.request(prompt, model, parse_mqm_answer, cache=cache, max_tokens=max_tokens)
            answers += parsed_answers
        return answers
