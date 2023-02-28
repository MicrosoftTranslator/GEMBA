# deployment is used to list available models
# for Azure API, specify model name as a key and deployment name as a value
# for OpenAI API, specify model name as a key and a value
credentials = {
    "deployments": {"text-davinci-002": "text-davinci-002"},
    "api_base": "https://******.openai.azure.com/",
    "api_key": "********************************",
    "requests_per_second_limit": 1
}