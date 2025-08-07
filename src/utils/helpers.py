import datetime
import json
import logging
import pprint
import re
import time
import unicodedata
from functools import partial
from itertools import chain
from multiprocessing import Pool, freeze_support
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from urllib.parse import urlparse

import html2text
import jsonlines
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from google import genai
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from torchmetrics.functional.text import bleu_score, rouge_score
from tqdm.auto import tqdm

# logging.basicConfig(level=logging.ERROR)
GEMINI_API_KEYS = [
    "AIzaSyBhiRySo93kKL_kRccEVY7SGNyyuWIFkEo",  # hoangtnm.cse@gmail.com
    "AIzaSyB-dJkIpMzsK-1VTL7d-KUMTqVF0-VRiz8",  # minhhoangtrannhat@gmail.com
    "AIzaSyAEeoUS9hQOFWE5uC4j-6mbu5QiXL1lyT8",  # hoangtnm.binance.sub1@gmail.com
    "AIzaSyA9_RHv3L6Wp-sQPeS-55e8SkM-XvHHlcY",  # hoangtnm.binance.sub2@gmail.com
    "AIzaSyCFsoJZZiHp_m6oSp81XFn9thc18HaDgrw",  # hoangtnm@d-soft.com.vn
]


def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))


def read_prompt(filepath: str) -> str:
    with open(filepath, "r") as f:
        prompt = f.readlines()
        return "".join(prompt)


def read_jsonl(filepath: Union[Path, str]) -> List[Dict[str, Any]]:
    with jsonlines.open(filepath) as reader:
        return [obj for obj in reader]


def write_jsonl(data, filepath: Union[Path, str]) -> None:
    with jsonlines.open(filepath, "w") as f:
        f.write_all(data)


def write_json(data: Union[Dict[str, Any], List[Any]], filepath: Union[Path, str]) -> None:
    with open(filepath, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def read_json(filepath: Union[Path, str]) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as fp:
        return json.load(fp)


def get_webdriver(headless: bool = True, implicit_wait: float = 5.0, verbose: bool = True) -> WebDriver:
    def get_options(headless: bool):
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
        return options

    try:
        options = get_options(headless)
        # options.binary_location = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        # print(e)
        if verbose:
            print("Fallback to headless mode...")
        options = get_options(headless=True)
        driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(implicit_wait)
    return driver


def get_soup_from_driver(driver: WebDriver) -> BeautifulSoup:
    return BeautifulSoup(driver.page_source, "html.parser")


def get_soup_from_url(url: str, retries: int = 5) -> Union[BeautifulSoup, None]:
    for retry in range(retries):
        if retry > 0:
            print(f"Retrying {url}... (Attempt {retry + 1})")

        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        if soup.body:
            return soup
    return


def find_element(driver: WebDriver, by: str, value: str, timeout: float = 5) -> WebElement:
    start = datetime.datetime.now()
    try:
        return WebDriverWait(driver, timeout).until(EC.presence_of_element_located((by, value)))
    except TimeoutException:
        end = datetime.datetime.now()
        elapsed = end - start
        print(f"TimeoutException: {by} {value} - elapsed {elapsed.total_seconds()}")
        return None


def find_elements(driver: WebDriver, by: str, value: str, timeout: float = 5) -> List[WebElement]:
    try:
        return WebDriverWait(driver, timeout).until(EC.presence_of_all_elements_located((by, value)))
    except TimeoutException:
        print(f"TimeoutException: {by} {value}")
        return []


def fetch_and_cache_html(args: Tuple[str, Path]) -> str:
    url, filepath = args
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    else:
        driver = get_webdriver()
        driver.get(url)
        time.sleep(0.5)
        page_source = driver.page_source
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(page_source)
            driver.quit()
            return page_source


def download_htmls(urls, filepaths, num_processes: int = 8):
    with Pool(processes=num_processes) as pool:
        _ = [x for x in pool.imap_unordered(fetch_and_cache_html, list(zip(urls, filepaths)))]


def create_html_parser():
    html_parser = html2text.HTML2Text()
    html_parser.body_width = 0
    html_parser.single_line_break = True
    html_parser.ignore_emphasis = True
    html_parser.mark_code = True
    return html_parser


class BaseCrawler:
    def __init__(self, url: str, headless: bool = True, implicit_wait: int = 10, init_driver: bool = True) -> None:
        self.url = url
        self.headless = headless
        self.implicit_wait = implicit_wait
        self.init_driver = init_driver

        if init_driver:
            self._init_driver(headless, implicit_wait, url)
        self.html_parser = html2text.HTML2Text()
        self.html_parser.body_width = 0
        self.html_parser.single_line_break = True
        self.html_parser.ignore_emphasis = True
        self.html_parser.mark_code = True

    def _init_driver(self, headless: bool, implicit_wait: int, url: str) -> None:
        self.driver = get_webdriver(headless, implicit_wait)
        self.driver.get(url)

    @property
    def base_url(self) -> str:
        parsed_url = urlparse(self.url)
        parent = str(Path(parsed_url.path).parent)
        parent = parent if parent != "/" else ""
        # return f"{parsed_url.scheme}://{parsed_url.netloc}{Path(parsed_url.path).parent}"
        return f"{parsed_url.scheme}://{parsed_url.netloc}{parent}"

    def cache_html(self, url: str, filepath: Path) -> str:
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        else:
            driver = get_webdriver()
            driver.get(url)
            time.sleep(0.5)
            page_source = driver.page_source
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(page_source)
                driver.quit()
                return page_source

    def run(self) -> None:
        raise NotImplementedError

    def cleanup(self) -> None:
        if self.init_driver:
            self.driver.quit()

    def _get_flatten_keys(self, x: Dict[str, List[str]]) -> List[str]:
        return list(x.keys()) + flatten_chain(x.values())

    def _fix_key(self, key: str, lower: bool = True) -> str:
        key = key.strip()
        key = key.replace(":", "")
        key = key.replace("\n", "")
        return key.lower() if lower else key

    def _get_key_by_value(self, data: Dict[str, List[str]], value: str) -> str:
        for key, values in data.items():
            if value in values:
                return key
        return value

    def get_element_level(self, element) -> int:
        level = 0
        while element.parent is not None:
            level += 1
            element = element.parent
        return level

    def remove_unicode_characters(self, text):
        text = re.sub(r"^[^a-zA-Z0-9\[]*", "", text)
        text = text.replace("\u2013", "-")
        text = text.replace("\u2014", " - ")
        for code in ["\u00a0", "\u202f"]:
            text = text.replace(code, " ")
        for code in ["\u201c", "\u201d"]:
            text = text.replace(code, '"')
        for code in ["\u2018", "\u2019"]:
            text = text.replace(code, "'")
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def compute_similarity(
    source: str,
    target: str,
    do_cos_sim: bool = False,
    n_gram: int = 4,
) -> Dict[str, float]:
    results = {}
    # if do_cos_sim:
    #     # https://huggingface.co/tasks/sentence-similarity
    #     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    #     embedding_1 = model.encode(source, convert_to_tensor=True)
    #     embedding_2 = model.encode(target, convert_to_tensor=True)
    #     cos_sim = sentence_transformers.util.pytorch_cos_sim(embedding_1, embedding_2).item()
    #     results["cos_sim"] = cos_sim
    results.update(
        {
            "bleu": bleu_score([source], [[target]], n_gram=n_gram, smooth=True).item(),
            "rougeL": rouge_score(
                source,
                target,
                rouge_keys="rougeL",
            )["rougeL_fmeasure"].item(),
        }
    )
    return results


def generate_content(api_keys, prompt, model="gemini"):
    api_key = np.random.choice(api_keys)
    client = genai.Client(api_key=api_key)
    respone = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return respone.text


def split_batch_inputs(batch_inputs: List[Dict[str, Any]], num_shards: int = 1):
    batch_inputs_shards = np.array_split(batch_inputs, num_shards)
    batch_inputs_shards = [i.tolist() for i in batch_inputs_shards]
    return batch_inputs_shards
