import ipdb
import json
import re
from collections import defaultdict


def esa_fewshot(few_shots):
    prompts = [
        {
            "role": "system",
            "content": f"Your task is to identify machine translation errors and assess the quality of the translation."
        }
    ]

    template = """{source_lang} source:
```{source_seg}```
{target_lang} translation:
```{target_seg}```

Based on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of two categories: major or minor. Major errors disrupt the flow and make the understandability of text difficult or impossible. Minor errors are errors that do not disrupt the flow significantly and what the text is trying to say is still understandable."""

    for shot in few_shots:
        prompts.append({
            "role": "user",
            "content": template.format(**shot)
        })
        answer = shot['answer']

        prompts.append({
            "role": "assistant",
            "content": answer
        })

    prompts.append({
            "role": "user",
            "content": template
        })

    return prompts


esa_few_shots = {
    "ende": {
            "source_lang": "English",
            "source_seg": "I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.",
            "target_lang": "German",
            "target_seg": "Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.",
            "answer": """Major:
accuracy/mistranslation - "involvement"
accuracy/omission - "the account holder"
Minor:
fluency/grammar - "wäre"
fluency/register - "dir"
""",
        },
    "encs": {
            "source_lang": "English",
            "source_seg": "Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.",
            "target_lang": "Czech",
            "target_seg": "Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemž obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.",
            "answer": """Major:
accuracy/addition - "ve Vídni"
accuracy/omission - "the stop-start"
Minor:
terminology/inappropriate for context - "partaje"
""",
        },
    "zhen": {
            "source_lang": "Chinese",
            "source_seg": "大众点评乌鲁木齐家居卖场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评",
            "target_lang": "English",
            "target_seg": "Urumqi Home Furnishing Store Channel provides you with the latest business information such as the address, telephone number, business hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.",
            "answer": """Major:
accuracy/addition - "of high-speed rail"
accuracy/mistranslation - "go to the reviews"
Minor:
style/awkward - "etc.,"
""",
        },
}

TEMPLATE_GEMBA_ESA_ERROR_SPANS = esa_fewshot([esa_few_shots['ende'], esa_few_shots['encs'], esa_few_shots['zhen']])

TEMPLATE_GEMBA_ESA_RANKING = 'Given the translation from {source_lang} to {target_lang} and the annotated error spans, assign a score on a continuous scale from 0 to 100. The scale has following reference points: 0="No meaning preserved", 33="Some meaning preserved", 66="Most meaning preserved and few grammar mistakes", up to 100="Perfect meaning and grammar".\n\nScore the following translation from {source_lang} source:\n```{source_seg}```\n{target_lang} translation:\n```{target_seg}```\nAnnotated error spans:\n```{error_spans}```\nScore (0-100): '
