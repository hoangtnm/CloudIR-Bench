import re
import time
import unicodedata
from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, element
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from tqdm.auto import tqdm, trange

from utils import (
    BaseCrawler,
    create_html_parser,
    find_element,
    flatten_chain,
    get_soup_from_driver,
    write_json,
)


def get_key_by_value(data, value):
    for key, values in data.items():
        if value in values:
            return key
    return value


class AzureCrawler(BaseCrawler):
    def __init__(self, url, headless, out_dir, implicit_wait=10, init_driver=True):
        super().__init__(url, headless, implicit_wait, init_driver)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_html_dir = self.out_dir / "html"
        self.out_html_dir.mkdir(exist_ok=True)
        self.html_parser = create_html_parser()

        self.data = []
        self.key_mapping = {
            "summary": [
                "what happened?",
                "summary ofimpact",
                "summary of impact",
                "impact statement",
            ],
            "root cause": [
                "root cause (updated 27 apr 2021)",
                "preliminary root cause",
                "summary root cause",
                "what went wrong and why?",  # dobule-check
                "what went wrong, and why?",  # dobule-check
            ],
            "mitigation": [
                "how did we respond?",
                "how did microsoft respond?",
            ],
            "next steps": [
                "next steps (updated 27 apr 2021)",
                "what happens next?",
            ],
            # "what went wrong and why?": ["what went wrong, and why?"],
            # "how did we respond?": ["how did microsoft respond?"],
            "how are we making incidents like this less likely or less impactful?": [
                "how are we making incidents like this less likely and less impactful?",
                "how we are making incidents like this less likely or less impactful?",
                "how is microsoft making incidents like this less likely, or at least less impactful?",
                "how are we making outages like this less likely or less impactful?",
            ],
            "feedback": [
                "how can we make our incident communication more useful?",
                "how can we make our incident communications more useful?",
                "provide feedback",
                "providefeedback",
            ],
            "how can our customers and partners make incidents like this less impactful?": [
                "how can customers and partners make incidents like this less impactful?",
            ],
            "how can customers make incidents like this less impactful?": [
                "how can customers make incidents like this less impactful"
            ],
        }
        self.known_keys = list(self.key_mapping.keys()) + flatten_chain(self.key_mapping.values())

    def _previous_page(self) -> None:
        prev_nav_btn = find_element(
            self.driver,
            By.CSS_SELECTOR,
            'div.wa-pagination > ul > li > a[aria-label="Previous page"]',
        )
        if prev_nav_btn:
            print("Going to previouse page...")
            prev_nav_btn.click()
            time.sleep(3)

    def _next_page(self) -> None:
        next_nav_btn = find_element(
            self.driver,
            By.CSS_SELECTOR,
            'div.wa-pagination > ul > li > a[aria-label="Next page"]',
        )
        if next_nav_btn:
            print("Going to next page...")
            next_nav_btn.click()
            time.sleep(3)

    @property
    def _current_page(self) -> str:
        return find_element(
            self.driver,
            By.CSS_SELECTOR,
            'div.wa-pagination > ul > li > a[aria-current="true"]',
        ).text.strip()

    def run(self) -> None:
        track_ids_filepath = self.out_dir / "track_ids.txt"
        if track_ids_filepath.exists():
            with open(track_ids_filepath, "r", encoding="utf-8") as f:
                track_ids = f.readlines()
                track_ids = [i.strip() for i in track_ids]
        else:
            self._init_driver(self.headless, self.implicit_wait, self.url)
            date_select = Select(find_element(self.driver, By.ID, "wa-dropdown-date"))
            date_select.select_by_value("all")
            time.sleep(3)

            track_ids = []
            soup = get_soup_from_driver(self.driver)
            page_nav = soup.select("div.wa-pagination > ul > li")
            num_page_end = int(page_nav[-2].text)

            for num_page in trange(1, num_page_end + 1):
                incidents = soup.select("div.incident-history-item")

                if num_page > 1:
                    self._next_page()

                while True:
                    soup = get_soup_from_driver(self.driver)
                    incidents = soup.select("div.incident-history-item")
                    if not incidents:
                        print("Cannot find any incident, retrying...")
                        self._previous_page()
                        self._next_page()
                    else:
                        break
                print(f"Crawling page {self._current_page}...")

                for incident in incidents:
                    incident_data = self.extract_incident_report(incident)
                    track_id = incident_data["track_id"]
                    with open(self.out_html_dir / f"{track_id}.html", "w", encoding="utf-8") as f:
                        f.write(str(incident))

                    track_ids.append(track_id)
                    self.data.append(incident_data)

            with open(self.out_dir / "track_ids.txt", "w", encoding="utf-8") as f:
                f.writelines([f"{track_id}\n" for track_id in track_ids])

        for track_id in tqdm(track_ids):
            # 0TK3-HPZ 04/30/2021
            # 8NVQ-HD8 03/09/2021
            # PLWV-BT0 02/26/2021: noise Summary of Impact
            # ZN8_-VT8 02/16/2021
            # CVTV-R80 02/12/2021: noise Provide feedback
            # 8KND-JP8 10/06/2020
            # CLCK-LD0 07/01/2020: noise Provide feedback
            # CT05-PC0 02/22/2020
            with open(self.out_html_dir / f"{track_id}.html", "r", encoding="utf-8") as f:
                # if not track_id == "0TK3-HPZ":
                # if not track_id == "VN11-JD8":
                # if not track_id == "PLWV-BT0":
                # if not track_id == "4L44-3F0":
                # if not track_id == "ZN7Y-5DG":
                # if not track_id == "8TY8-HT0":
                # if not track_id == "ZN8_-VT8":
                #     continue
                incident = BeautifulSoup(f.read(), "html.parser")
                incident_data = self.extract_incident_report(incident)
                self.data.append(incident_data)

        keys = [x.keys() for x in self.data]
        columns, count = np.unique(flatten_chain(keys), return_counts=True)

        write_json(self.data, self.out_dir / "all_data.json")
        df = pd.DataFrame(self.data)
        # df = df.drop(columns="raw_text")
        with pd.ExcelWriter(self.out_dir / "all_data.xlsx") as writer:
            df.to_excel(writer, sheet_name="main", index=False)
            x = pd.DataFrame({"columns": columns, "count": count})
            x.to_excel(writer, sheet_name="stats", index=False)
            # x = df.drop_duplicates(["title", "summary"]).copy()
            x = df.drop_duplicates(["title"]).copy()
            print(df.shape, x.shape)

        # stats = []
        # for col in df.columns:
        #     df[col] = df[col].str.count(" ") + 1
        #     stats.append({"name": col, "num_values": df[col].count(), "num_words_avg": df[col].mean()})
        # df_stats = pd.DataFrame(stats)
        # print(df_stats)

    def extract_incident_report(self, incident):
        date = incident.select_one("span.hide-text").text
        title = incident.select_one("div.incident-history-title").text.strip()
        track_id = incident.select_one("div.incident-history-tracking-id").text.strip()
        track_id = track_id.split(": ")[1]

        # TODO
        for x in incident.find_all(
            # lambda tag: tag.name in ["strong", "span", "br"] and
            lambda tag: self.remove_unicode_characters(tag.text.strip().replace(":", "")) == ""
        ):
            x.decompose()

        body = incident.select_one("div.row div.card-body")
        raw_text = self.remove_unicode_characters(body.text.strip())
        incident_data = {
            "date": date,
            "title": title.replace("\u2013", "-"),
            "track_id": track_id,
            # "raw_text": raw_text,
        }
        # prev_key, current_key = None, None
        # for idx, ele in enumerate(body.find_all(recursive=False)):
        #     ele_text = ele.text.strip()

        #     # Skip: Watch our 'Azure Incident Retrospective' video about this incident
        #     if idx == 0 and ele.select_one("em"):
        #         continue

        #     if ele.name == "strong":
        #         current_key = self._fix_key(ele.text)
        #         if track_id == "8TY8-HT0":
        #             print(current_key)
        #         current_key = get_key_by_value(self.key_mapping, current_key)
        #         prev_key = current_key

        #         if not incident_data.get(prev_key):
        #             incident_data[prev_key] = ""
        #             # print(f"prev_key: {prev_key}, level: {self.get_element_level(ele)}")

        #         next_sibling = ele.next_sibling
        #         while next_sibling and next_sibling.name != "strong":
        #             value = next_sibling.text.strip() if next_sibling else ""
        #             value = self.remove_unicode_characters(f"{value}") if not next_sibling.name == "a" else f"{value}"
        #             # value = f"{value} " if next_sibling.name == "a" else f"{value} "
        #             incident_data[prev_key] += f"{value} "
        #             # print(self.get_element_level(next_sibling), "...", value)

        #             next_sibling = next_sibling.next_sibling
        #             if next_sibling and not isinstance(next_sibling, str):
        #                 if next_sibling.name == "strong" or next_sibling.find("strong"):
        #                     break

        #     elif ele.name != "ul" and ele.find("strong", recursive=False):
        #         for sub_ele in ele.find_all(recursive=False):
        #             sub_ele_text = sub_ele.text.strip()

        #             if sub_ele.name == "strong":
        #                 current_key = self._fix_key(sub_ele.text, lower=False)
        #                 if track_id == "VN11-JD8" and current_key.lower() not in self.known_keys:
        #                     current_key = self._fix_key(current_key, lower=False)
        #                     incident_data[prev_key] += f"{current_key}\n"
        #                 else:
        #                     current_key = self._fix_key(current_key)
        #                     prev_key = current_key
        #                 prev_key = get_key_by_value(self.key_mapping, prev_key)

        #                 if not incident_data.get(prev_key):
        #                     incident_data[prev_key] = ""

        #                 next_sibling: element.NavigableString = sub_ele.next_sibling
        #                 while next_sibling and next_sibling.name != "strong":
        #                     value = next_sibling.text.strip() if next_sibling else ""
        #                     value = (
        #                         self.remove_unicode_characters(f"{value}")
        #                         if not next_sibling.name == "a"
        #                         else f"{value}"
        #                     )
        #                     # value = f"{value} " if next_sibling.name == "a" else f"{value} "
        #                     incident_data[prev_key] += f"{value} "

        #                     next_sibling = next_sibling.next_sibling
        #                     if next_sibling and not isinstance(next_sibling, str):
        #                         if next_sibling.name == "strong" or next_sibling.find("strong"):
        #                             break

        #     elif ele.name == "a":
        #         continue

        #     elif prev_key and ele.name != "ul" and not ele.find("strong"):
        #         incident_data[prev_key] += f"\n{self.remove_unicode_characters(ele_text)}\n"

        #     elif ele.name == "ul":
        #         ele_text = [x.text.strip() for x in ele.find_all(recursive=False)]
        #         ele_text = [self.remove_unicode_characters(x) for x in ele_text]
        #         ele_text = "\n".join([f"- {x}" for x in ele_text])
        #         incident_data[prev_key] += f"\n{ele_text}\n"

        #     # num_strong = len(ele.find_all("strong", recursive=False))
        #     # if num_strong:
        #     #     for strong_tag in ele.find_all('strong', recursive=False):
        #     #         current_key = strong_tag.get_text(strip=True).replace(":", "").lower()
        #     #         current_key = current_key.replace("\n", " ")
        #     #         if track_id == "VN11-JD8" and current_key not in self.known_keys:
        #     #             incident_data[prev_key] += f"{current_key}\n"
        #     #         else:
        #     #             prev_key = current_key

        #     #         prev_key = get_key_by_value(self.key_dict, prev_key)
        #     #         if not incident_data.get(prev_key):
        #     #             incident_data[prev_key] = ""
        #     #         # prev_key = current_key

        #     #         # Find the next sibling (excluding <br>) and get its text
        #     #         next_sibling = strong_tag.next_sibling
        #     #         while next_sibling and next_sibling.name != "strong":
        #     #             if next_sibling.name == "br":
        #     #                 next_sibling = next_sibling.next_sibling
        #     #                 continue

        #     #             value = next_sibling.get_text(strip=True) if next_sibling else ""
        #     #             # value = value.replace(u"\u2013", "-")
        #     #             value = remove_unicode_characters(f"{value}\n") if not next_sibling.name == "a" else f"{value} "
        #     #             incident_data[prev_key] += value
        #     #             next_sibling = next_sibling.next_sibling

        #     # else:
        #     #     if ele.name == "ul":
        #     #         ele_text = "\n".join([x.text.strip() for x in ele.find_all(recursive=False)])
        #     #     if prev_key:
        #     #         incident_data[prev_key] += f"{remove_unicode_characters(ele_text)}\n"

        #     # if ele.select_one("strong") and ele.name != "ul":
        #     #     # TODO:
        #     #     # if ele.select_one("br"):
        #     #     #     print(track_id, "...")

        #     #     x = ele.select_one("strong").text.strip().replace("\n", " ")

        #     #     if x == ele_text and x.endswith("?"):
        #     #         current_key = x.lower()
        #     #         # current_key = self.key_dict.get(current_key, current_key)
        #     #         current_key = get_key_by_value(self.key_dict, current_key)
        #     #         incident_data[current_key] = ""
        #     #         continue
        #     #     elif x.endswith(":") and x != ele_text:
        #     #         tmp = ele_text.split(":")
        #     #         current_key = tmp[0].strip().replace("\n", " ").lower()
        #     #         # current_key = self.key_dict.get(current_key, current_key)
        #     #         current_key = get_key_by_value(self.key_dict, current_key)
        #     #         ele_text = ":".join(tmp[1:]).strip()
        #     #         incident_data[current_key] = ""

        #     # # Handles list
        #     # if ele.name == "ul":
        #     #     ele_text = "\n".join([x.text.strip() for x in ele.find_all(recursive=False)])

        #     # if current_key:
        #     #     incident_data[current_key] += f"{remove_unicode_characters(ele_text)}\n"

        # if incident_data.get("root cause and mitigation"):
        #     incident_data["root cause"] = incident_data["root cause and mitigation"]
        #     incident_data["mitigation"] = incident_data["root cause and mitigation"]
        #     incident_data.pop("root cause and mitigation")
        include_keys = ["date", "title", "track_id", "title", "summary", "root cause", "mitigation"]
        incident_data = {k: v.strip() for k, v in incident_data.items() if k in include_keys}
        # raw_text = ""
        # for k, v in incident_data.items():
        #     raw_text += f"{k}: {v}\n\n"
        incident_data["human_labels"] = {}
        incident_data["html"] = self.remove_unicode_characters(self.html_parser.handle(str(body)))
        incident_data["start_timezone"] = ""
        incident_data["start_time"] = ""
        incident_data["end_timezone"] = ""
        incident_data["end_time"] = ""
        return incident_data


def main():
    parser = ArgumentParser()
    parser.add_argument("--url", default="https://azure.status.microsoft/en-us/status/history")
    parser.add_argument("-o", "--out_dir", required=True)
    args = parser.parse_args()

    crawler = AzureCrawler(args.url, out_dir=args.out_dir, headless=True, init_driver=False)
    crawler.run()
    crawler.cleanup()


if __name__ == "__main__":
    main()
