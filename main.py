import logging
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def introduction_main(inp):
    """
    Функция, которая принимает на вход датафрейм с полем 'text', выделяет из него вступительную часть, основную часть
    и записывает вступительную часть в поле 'introduction'
    """
    logger.info("Загрузка датафрейма: %s", inp)
    df = pd.read_csv(inp, low_memory=False)  # Загрузка датафрейма

    logger.info("Выделение вступительной части из сообщения")
    df["introduction"] = np.vectorize(transform)(df["text"]) # Вписываем в 'introduction' вступительную часть

    return df

def transform(message):
    """
    Функция, в которой проводится предобработка сообщения, его токенезация и выделение частей
    """
    logger.debug("Токенизация сообщения: %s", message)

    # Проверяем наличие переноса строки в сообщении
    if "\n" in message:
        logger.debug("Перенос строки найден в сообщении")
        return message.split("\n")[0].strip()  # Если в сообщении есть перенос строки, то мы считаем текст до первого переноса как вступительную часть

    vectorizer = TfidfVectorizer()  # Загружаем tf-idf преобразователь для определения наиболее важных слов
    tfidf_matrix = vectorizer.fit_transform([message])

    nlp = spacy.load("en_core_web_sm")  # Загружаем модель SpaCy для автоматической предобработки и токенезации сообщения
    nlp.max_length = 20000000  # Увеличиваем число слов в сообщении
    doc = nlp(message)  # Проводим токенезацию, лемматизацию и определение частей речи в сообщении

    # Создаем словарь наиболее значимых слов в сообщении
    feat_names = vectorizer.get_feature_names_out()
    feat_index = tfidf_matrix.nonzero()[1]
    tfidf_scores = zip(feat_index, [tfidf_matrix[0, x] for x in feat_index])
    important_words = sorted([(feat_names[idx], score) for idx, score in tfidf_scores], key=lambda x: x[1], reverse=True)

    # Выделяем вступительную часть как текст до первого значимого глагола
    introduction_part = ""
    verb_found = False  # Флаг для отслеживания наличия значимого глагола в сообщении
    for token in doc:
        if token.text.lower() in [word for word, _ in important_words]:
            if token.pos_ == "VERB":
                verb_found = True
                break
            introduction_part += token.text_with_ws

    # Если вступительная часть пустая и значимый глагол не найден, записываем None
    if not introduction_part:
        introduction_part = "None"

    return introduction_part.strip()