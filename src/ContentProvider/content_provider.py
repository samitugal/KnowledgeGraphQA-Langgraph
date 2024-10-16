import json
import os
import random
import string
import zipfile
from dataclasses import dataclass
from typing import List, Tuple

from dotenv import load_dotenv

from ..output_models import Answer, Article, Paragraph, QuestionAnswer

load_dotenv()


class ContentProvider:
    def __init__(self):
        self.path = os.environ["KAGGLE_DATASET_PATH"]
        content = self.get_content()
        self.all_contexts, self.all_qas = self.collect_contexts_and_qas(content)

    def _parse_article(self, data) -> Article:
        paragraphs = []
        for p in data["paragraphs"]:
            qas = []
            for qa in p["qas"]:
                answers = [Answer(**ans) for ans in qa["answers"]]
                qas.append(
                    QuestionAnswer(
                        answers=answers, question=qa["question"], id=qa["id"]
                    )
                )
            paragraphs.append(Paragraph(context=p["context"], qas=qas))
        return Article(title=data["title"], paragraphs=paragraphs)

    def get_content(self, mock: bool = False) -> Article:
        with open(os.path.join(self.path, "dev-v1.1.json")) as f:
            dev_data = json.load(f)

        if mock:
            return self._parse_article(dev_data["data"][0])
        else:
            return self._parse_article(
                dev_data["data"][random.randint(0, len(dev_data["data"]) - 1)]
            )

    def collect_contexts_and_qas(self, article: Article) -> Tuple[str, List[dict]]:
        all_contexts = ""
        all_qas = []

        for paragraph in article.paragraphs:
            all_contexts += paragraph.context + " "
            for qa in paragraph.qas:
                all_qas.append(
                    {
                        "question": qa.question,
                        "answers": [answer.text for answer in qa.answers],
                    }
                )

        all_contexts = all_contexts.strip()
        translator = str.maketrans("", "", string.punctuation)
        all_contexts = all_contexts.translate(translator)
        return all_contexts, all_qas
