import openai
import time
from termcolor import colored
from datetime import datetime


# class for calling OpenAI API and handling cache
class GptApi:
    def __init__(self, credentials, verbose=True):
        assert "api_key" in credentials, "api_key not found in credentials"
        assert "deployments" in credentials, "deployments not found in credentials"

        self.deployments = credentials["deployments"]
        self.verbose = verbose

        if "api_base" in credentials:
            # Azure API access
            openai.api_type = "azure"
            openai.api_version = "2022-06-01-preview"
            openai.api_base = credentials["api_base"]
            openai.api_key = credentials["api_key"]
            self.api_type = "azure"
        else:
            # OpenAI API access
            openai.api_key = credentials["api_key"]
            self.api_type = "openai"

        # limit the number of requests per second
        if "requests_per_second_limit" in credentials:
            self.rps_limit = 1 / credentials["requests_per_second_limit"]
        else:
            self.rps_limit = 0
        self.last_call_timestamp = 0

    # answer_id is used for determining if it was the top answer or how deep in the list it was
    def request(self, prompt, model, parse_response, temperature=0, answer_id=-1, cache=None):
        max_tokens = 20
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
                print(f"Answer (t={temperature}): " + colored(answer, "yellow") + " (" + colored(full_answer, "blue") + ")")
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

        if max_tokens > 500 or temperature > 10:
            return []

        dt = datetime.now()
        ts = datetime.timestamp(dt)
        if ts - self.last_call_timestamp < self.rps_limit:
            time.sleep(self.rps_limit - (ts - self.last_call_timestamp))

        self.last_call_timestamp = ts

        if self.verbose:
            print(prompt)
        while True:
            try:
                response = self.call_api(prompt, model, n, temperature, max_tokens)
                break
            except Exception as e:
                # response was filtered
                if hasattr(e, 'code'):
                    if e.code == 'content_filter':
                        return []
                    print(e.code)
                # frequent error is reaching the API limit
                print(colored("Error, retrying...", "red"))
                print(e)
                time.sleep(1)

        answers = []
        for choice in response["choices"]:
            answer = choice['text'].strip()
            # one of the responses didn't finish, we need to request more tokens
            if choice["finish_reason"] != "stop" and model != "text-chat-davinci-002":  # TODO remove exception
                if self.verbose:
                    print(colored(f"Increasing max tokens to fit answers.", "red") + colored(answer, "blue"))
                return self.request_api(prompt, model, temperature=temperature, max_tokens=max_tokens + 200)

            answers.append({
                "answer": answer,
                "finish_reason": choice["finish_reason"],
            })

        if len(answers) > 1:
            # remove duplicate answers
            answers = [dict(t) for t in {tuple(d.items()) for d in answers}]

        return answers

    def call_api(self, prompt, model, n, temperature, max_tokens):
        if self.api_type == "azure":
            return openai.Completion.create(
                engine=self.deployments[model],
                prompt=prompt,
                temperature=temperature/10,
                max_tokens=max_tokens,
                top_p=1,
                n=n,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
        else:
            return openai.Completion.create(
                model=self.deployments[model],
                prompt=prompt,
                temperature=temperature/10,
                max_tokens=max_tokens,
                top_p=1,
                n=n,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
