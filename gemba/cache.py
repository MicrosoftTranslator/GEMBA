import os
import json
import time


# this class is used to cache answers from API calls
# it is multithread save, but suboptimal as it doesn't use the latest data version
# therefore duplicate API calls might be made
class Cache:
    def __init__(self, cache_file):
        self.folder = "cache"
        self.cache_filename = cache_file
        self.cache_file = f"{self.folder}/{cache_file}"
        self.last_reload = 0
        self.RELOAD_INTERVAL = 5 * 60  # seconds
        self.cache = {}
        # appending to a long file line by line is slow, batch it
        self.to_batch_append = []
        self._load_cache()

    def __del__(self):
        if len(self.to_batch_append) > 0:
            self._load_cache(True)

    def _load_cache(self, force=False):
        # don't reload cache too often
        if self.last_reload > time.time() - self.RELOAD_INTERVAL and not force:
            return

        self.cache = {}
        if os.path.exists(self.cache_file):
            data = []
            with open(self.cache_file, "r", encoding="utf-8") as f:
                # read json lines file
                for line in f:
                    data.append(line)
            for d in data:
                self._add_to_cache(json.loads(d))
            self.last_reload = time.time()
        else:
            # create file if it doesn't exist with all necessary folders
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

        if len(self.to_batch_append) > 0:
            for d in self.to_batch_append:
                self._add_to_cache(d)
            self._save_cache()
            self.to_batch_append = []

    def _save_cache(self):
        print(f"Saving cache {self.cache_filename}")
        filename = f"{self.folder}/tmp_{self.cache_filename}.cache"
        with open(filename, "w", encoding='utf8') as f:
            for model in self.cache:
                for temperature in self.cache[model]:
                    for prompt in self.cache[model][temperature]:
                        answers = self.cache[model][temperature][prompt]
                        if len(answers) > 1:
                            # remove duplicate answers
                            answers = [dict(t) for t in {tuple(d.items()) for d in answers}]
                        data = {
                            "model": model,
                            "temperature": temperature,
                            "answers": answers,
                            "prompt": prompt,
                        }
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        # move tmp.cache file to self.cache_file
        os.replace(filename, self.cache_file)

    def _add_to_cache(self, data):
        model = data['model']
        temperature = data['temperature']
        prompt = data['prompt']
        answers = data['answers']

        # if prompt is dictionary or list, convert it to json
        if type(prompt) == dict or type(prompt) == list:
            prompt = json.dumps(prompt, ensure_ascii=False)

        if len(answers) == 0:
            return

        if model not in self.cache:
            self.cache[model] = {}
        if temperature not in self.cache[model]:
            self.cache[model][temperature] = {}

        self.cache[model][temperature][prompt] = answers

    def get(self, data):
        model = data['model']
        temperature = data['temperature']
        prompt = data['prompt']
        
        # if prompt is dictionary or list, convert it to json
        if type(prompt) == dict or type(prompt) == list:
            prompt = json.dumps(prompt, ensure_ascii=False)

        if model in self.cache and \
           temperature in self.cache[model] and \
           prompt in self.cache[model][temperature]:
            return self.cache[model][temperature][prompt]

        return None

    def add(self, data):
        self._add_to_cache(data)
        self.to_batch_append.append(data)
        # reload/save cache in case of multithreading
        self._load_cache()
