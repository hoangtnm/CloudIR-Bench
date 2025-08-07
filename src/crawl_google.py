import re
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Union

import html2text
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, element
from tqdm.auto import tqdm
from utils import (
    BaseCrawler,
    flatten_chain,
    get_soup_from_driver,
    get_soup_from_url,
    get_webdriver,
    write_json,
)


def check_startswith_any(string, prefixes):
    """
    Checks if a given string starts with any of the strings in a list of prefixes.

    Args:
      string: The string to check.
      prefixes: A list of strings to check as prefixes.

    Returns:
      True if the string starts with any of the prefixes, False otherwise.
    """
    for prefix in prefixes:
        if string.startswith(prefix):
            return True
    return False


class GoogleCrawler(BaseCrawler):
    def __init__(self, platform, headless, out_dir, implicit_wait=10, init_driver=True):
        assert platform in {"cloud", "workspace"}
        if platform == "cloud":
            url = "https://status.cloud.google.com/summary"
        else:
            url = "https://www.google.com/appsstatus/dashboard/summary"

        super().__init__(url, headless, implicit_wait, init_driver)
        self.platform = platform
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.summary_filepath = self.out_dir / f"gg_{platform}_summary.csv"
        self.data = []
        self.h1, self.h2 = [], []
        self.h12_headers = []
        self.incidents_with_root_cause = []
        self.incidents_with_remediation = []
        self.key_mapping = {
            "incident report": [
                "updated incident report",
            ],
            "introduction": ["introduction"],
            "service(s) affected": ["service(s) affected"],
            "zone(s) affected": ["service(s) affected"],
            "summary": [
                "incident summary",
                "issue summary",
                "issue summary (all times in us/pacific daylight time)",
            ],
            "impact": [
                "description of impact",
                "detailed description of impact",
                "customer impact",
            ],
            "root cause and impact": ["root cause and impact"],
            "root cause": [
                "root cause and trigger",
                "root cause of device reboots",
            ],
            "mitigation": [
                "remediation",
                "remediation and prevention",
                "remediation and prevention/detection",
                "water leak, mitigation, and prevention",
                "mitigation & prevention of additional services",
            ],
            "root cause and mitigation": ["root cause and mitigation"],
            "sla credits": ["sla credits"],
            "background": ["background"],
            "how customers experienced the issue": ["how customers experienced the issue"],
            "workaround": ["workaround(s)"],
            "additional details": ["additional details"],
            "raw_text": ["raw_text"],
            "diagnosis": ["diagnosis"],
            "is_mini_report": ["is_mini_report"],
        }
        self.known_keys = self._get_flatten_keys(self.key_mapping)
        self.count = 0
        self.num_mini_report = 0
        self.ids_mini_report = []
        self.ids_report = []

    # def _extract_incident_report_v2(self, soup, incident_id: str):

    def _extract_incident_report(self, soup, incident_id: str):
        # if self.platform == "cloud":
        #     incident_header = soup.select_one("h2.incident-header").text.strip()
        # else:
        incident_header = soup.select_one(".incident-header").text.strip()
        start_time = soup.select_one("strong.start-time").text.strip()
        try:
            end_time = soup.select_one("strong.end-time").text.strip()
        except AttributeError as e:
            print(f"Cannot extract end_time for ID {incident_id}...")
            end_time = "N/A"

        try:
            incident_description = soup.select_one(".incident-description").text.strip()
        except AttributeError:
            incident_description = None

        try:
            locations = [i.text.strip() for i in soup.select("p.past-locations.locations > span")]
        except AttributeError:
            locations = None

        status_updates = soup.select("table.status-updates > tbody > tr")[::-1]
        updates = []
        parsed_html = ""
        for update_idx, update in enumerate(status_updates):
            status, date, time, description = update.find_all("td")
            status = status.find("svg")["aria-label"].strip().split()[0]
            date = date.text.strip()
            time = time.text.strip().upper()
            x = f"{date} {time}".split()
            timestamp = " ".join(x[:-1])
            timezone = x[-1]
            if self.platform == "cloud":
                timestamp = datetime.strptime(timestamp, "%d %b %Y %H:%M")
            else:
                for old_value, new_value in zip(["Sept", "June", "July"], ["Sep", "Jun", "Jul"]):
                    timestamp = timestamp.replace(old_value, new_value)
                timestamp = datetime.strptime(timestamp, "%d %b %Y %I:%M %p")
            update_time = timestamp.strftime("%H:%M")
            update_date = timestamp.day
            update_month = timestamp.strftime("%b")
            update_year = timestamp.strftime("%Y")
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M")
            update_time_str = f"{update_time} {timezone} on {update_date} {update_month} {update_year}"

            for x in description.find_all(
                lambda tag: tag.text.strip().lower() in ["incident report", "updated incident report"]
            ):
                x.decompose()
            for x in description.find_all(
                lambda tag: self.remove_unicode_characters(tag.text.strip().replace(":", "")) == ""
            ):
                # print("...", x)
                x.decompose()

            html_parser = html2text.HTML2Text()
            html_parser.body_width = 0
            html_parser.single_line_break = True
            html_parser.ignore_emphasis = True
            html_parser.mark_code = True
            update_message = html_parser.handle(str(description)).strip()
            update_message = self.remove_unicode_characters(update_message)
            update_message = update_message.replace("\n\n\n", "\n")
            update_message = update_message.replace("\n\n", "\n")
            message_with_status = f"[{status}] {update_time_str}:\n {update_message}\n\n"
            message_without_status = f"{update_time_str}:\n {update_message}\n\n"
            if update_idx in [0, len(status_updates) - 2, len(status_updates) - 1]:
                parsed_html += message_with_status
            else:
                parsed_html += message_without_status

            description_raw = description.text.strip()
            updates.append(
                {
                    "status": status,
                    "date": date,
                    "time": time,
                    "timestamp": timestamp,
                    "timezone": timezone,
                    "description_raw": description_raw,
                }
            )
        start_tz, end_tz = updates[0]["timezone"], updates[-1]["timezone"]
        # start_time, end_time = updates[0]["timestamp"], updates[-1]["timestamp"]
        # assert start_tz == end_tz, f"{incident_id}"

        # TODO: Extracts keys and values
        incident_data = {}
        latest_update = status_updates[-1]
        _, _, _, latest_description = latest_update.find_all("td")
        prev_key, current_key = None, None
        key_name = None
        # root_cause_ele = latest_description.find(["h1", "h2"], string=re.compile("^root cause", re.IGNORECASE))
        root_cause_ele = latest_description.find(["h1", "h2", "p"], string=re.compile("^root cause", re.IGNORECASE))
        # root_cause_ele = latest_description.find(["p"], string=re.compile("^root cause", re.IGNORECASE))

        if root_cause_ele:
            self.ids_report.append(incident_id)
            key_name = root_cause_ele.name
            self.count += 1

            ele: Union[element.NavigableString, element.Tag]
            # for ele in latest_description.find_all(key_name, recursive=False):
            for ele in latest_description.find_all(recursive=False):
                text = self._fix_key(ele.text, lower=True)
                text = self.remove_unicode_characters(text)
                text = self._get_key_by_value(self.key_mapping, text)
                if text in self.known_keys and text == prev_key:
                    # PCibvK1LbstDfPKPGfRC
                    # vLsxuKoRvykNHW3nnhsJ
                    incident_data[prev_key] += f"\n{ele.text}"
                elif text in self.known_keys:
                    prev_key = text
                    if not incident_data.get(prev_key):
                        incident_data[prev_key] = ""

                elif prev_key and ele.name not in {"ul", "ol"}:
                    ele_text = self.remove_unicode_characters(ele.text)
                    ele_text = ele_text.strip().replace("\n", " ")
                    # if ele.name.startswith("h"):
                    #     ele_text = f"# {ele_text}: "
                    incident_data[prev_key] += f"\n{ele_text}"

                elif prev_key and ele.name == "ul":
                    for ele_li in ele.find_all(recursive=False):
                        if ele_li.find("ul"):
                            incident_data[prev_key] += (
                                f"\n- {self.remove_unicode_characters(ele_li.contents[0].text.strip())}"
                            )
                            for sub_sub_ele in ele_li.find_all("li", recursive=True):
                                incident_data[prev_key] += f"\n\t- {self.remove_unicode_characters(sub_sub_ele.text)}"
                        else:
                            incident_data[prev_key] += f"\n- {self.remove_unicode_characters(ele_li.text)}"

                elif prev_key and ele.name == "ol":
                    for ele_idx, ele_li in enumerate(ele.find_all(recursive=False)):
                        if ele_li.find("ul"):
                            incident_data[prev_key] += (
                                f"\n{ele_idx + 1}. {self.remove_unicode_characters(ele_li.contents[0].text.strip())}"
                            )
                            for sub_sub_ele in ele_li.find_all("li", recursive=True):
                                incident_data[prev_key] += f"\n\t- {self.remove_unicode_characters(sub_sub_ele.text)}"
                        else:
                            incident_data[prev_key] += f"\n{ele_idx + 1}. {self.remove_unicode_characters(ele_li.text)}"

        if incident_id == "eCPQKkKcFy6NYXExnPXL":
            incident_data.update(
                {
                    "root cause": incident_data["root cause and mitigation"],
                    "mitigation": incident_data["root cause and mitigation"],
                }
            )
            incident_data.pop("root cause and mitigation")

        if incident_id == "kDBRnSgQCPw93E8vKKat":
            root_cause_impact = incident_data["root cause and impact"].split("\n")
            root_cause = "\n".join(root_cause_impact[:-1])
            impact = root_cause_impact[-1]
            incident_data.update({"root cause": root_cause, "impact": impact})
            incident_data.pop("root cause and impact")

        if incident_data.get("root cause"):
            incident_data.update(
                {
                    "summary": incident_data.get("summary", "Check URL for details"),
                    "impact": incident_data.get("impact", "Check URL for details"),
                    "mitigation": incident_data.get("mitigation", "Check URL for details"),
                }
            )

        if latest_update.find(["h1", "h2", "p"], string=re.compile("^root cause", re.IGNORECASE)):
            self.incidents_with_root_cause.append(incident_id)
        if latest_update.find(["h1", "h2", "p"], string=re.compile("^remediation", re.IGNORECASE)):
            self.incidents_with_remediation.append(incident_id)

        h12_headers = latest_update.find_all(["h1", "h2"])
        h12_headers = [i.text.lower().strip() for i in h12_headers]
        h1_headers = [i.text.lower().strip() for i in latest_update.select("h1")]
        h2_headers = [i.text.lower().strip() for i in latest_update.select("h2")]
        self.h1.extend(h1_headers)
        self.h2.extend(h2_headers)

        if latest_description.find(["h1", "h2", "p"], string=re.compile("mini incident report", re.IGNORECASE)):
            raw_text = ""
            for update_idx, update in enumerate(status_updates):
                # status, date, time, description = update.find_all("td")
                description = update.find_all("td")[-1]
                status = updates[update_idx]["status"]
                date = updates[update_idx]["date"]
                time = updates[update_idx]["time"]
                raw_text += f"# {status}: {date} - {time}"

                for ele in description.find_all(recursive=False):
                    text = self.remove_unicode_characters(ele.text)
                    if ele.name not in {"ul", "ol"}:
                        ele_text = self.remove_unicode_characters(ele.text)
                        ele_text = ele_text.strip().replace("\n", " ")
                        raw_text += f"\n{ele_text}"

                    elif ele.name == "ul":
                        for ele_li in ele.find_all(recursive=False):
                            if ele_li.find("ul"):
                                raw_text += f"\n- {self.remove_unicode_characters(ele_li.contents[0].text.strip())}"
                                for sub_sub_ele in ele_li.find_all("li", recursive=True):
                                    raw_text += f"\n\t- {self.remove_unicode_characters(sub_sub_ele.text)}"
                            else:
                                raw_text += f"\n- {self.remove_unicode_characters(ele_li.text)}"

                    elif ele.name == "ol":
                        for ele_idx, ele_li in enumerate(ele.find_all(recursive=False)):
                            if ele_li.find("ul"):
                                raw_text += f"\n{ele_idx + 1}. {self.remove_unicode_characters(ele_li.contents[0].text.strip())}"
                                for sub_sub_ele in ele_li.find_all("li", recursive=True):
                                    raw_text += f"\n\t- {self.remove_unicode_characters(sub_sub_ele.text)}"
                            else:
                                raw_text += f"\n{ele_idx + 1}. {self.remove_unicode_characters(ele_li.text)}"

                raw_text += "\n\n"

            self.num_mini_report += 1
            self.ids_mini_report.append(incident_id)
            incident_data.update(
                {
                    # "raw_text": raw_text,
                    "summary": "",
                    "root cause": "",
                    "mitigation": "",
                    "diagnosis": "",
                    "workaround": "",
                    "is_mini_report": "True",
                }
            )

        incident_data = {k: v.strip() for k, v in incident_data.items()}
        return {
            "title": incident_header,
            # "diagnosis": diagnosis_logs,
            "h1_headers": h1_headers,
            "h2_headers": h2_headers,
            "h12_headers": h12_headers,
            "description": incident_description,
            "locations": locations,
            # "updates": updates,
            **incident_data,
            "start_timezone": start_tz,
            "start_time": start_time,
            "end_timezone": end_tz,
            "end_time": end_time,
            "html": parsed_html,
        }

    def run(self) -> None:
        if not self.summary_filepath.exists():
            self._init_driver(self.headless, self.implicit_wait, self.url)
            incidents_list = []
            soup = get_soup_from_driver(self.driver)
            active_products = soup.select("div.active-products-category > psd-product-table")
            inactive_products = soup.select("div.inactive-products-category > psd-product-table")
            all_products = flatten_chain([active_products, inactive_products])
            for product in tqdm(all_products):

                def _selector_product_name(tag):
                    return tag.name == "span" and tag.has_attr("class") and "product-name" in tag.get("class")[0]

                def _selector_no_rows(tag):
                    return tag.name == "td" and tag.has_attr("class") and "no-rows-message" in tag.get("class")[0]

                product_name = product.find(_selector_product_name).text
                product_href = product.find(string="See more").parent.parent["href"]
                product_url = f"{self.base_url}/{product_href}"

                # product_url = "https://status.cloud.google.com/products/fPovtKbaWN9UTepMm3kJ/history"
                product_soup = get_soup_from_url(product_url)
                no_incidents = product_soup.find(_selector_no_rows)

                if no_incidents:
                    # print(f"No incidents for {product_name}: {product_url}")
                    pass
                else:
                    incidents = product_soup.select("psd-table-rows table > tbody > tr")
                    for incident in incidents:
                        summary, date, duration = [i.text.strip() for i in incident.find_all("td")]
                        incident_href = "/".join(incident.select_one("td a")["href"].split("/")[-2:])
                        incident_url = f"{self.base_url}/{incident_href}"
                        incident_id = Path(incident_href).stem
                        incident_html_filepath = self.out_dir / "html" / f"{incident_id}.html"

                        if incident_html_filepath.exists():
                            with open(incident_html_filepath, "r", encoding="utf-8") as f:
                                incident_page_source = f.read()
                        else:
                            print(f"Crawling {incident_url}")
                            incident_driver = get_webdriver()
                            incident_driver.implicitly_wait(5)
                            incident_driver.get(incident_url)
                            incident_page_source = incident_driver.page_source
                            incident_html_filepath.parent.mkdir(parents=True, exist_ok=True)
                            with open(incident_html_filepath, "w", encoding="utf-8") as f:
                                f.write(incident_page_source)
                            incident_driver.quit()

                        incident_soup = BeautifulSoup(incident_page_source, "html.parser")
                        incident_report = self._extract_incident_report(incident_soup, incident_id)
                        incident_filepath = self.out_dir / "json" / f"{incident_id}.json"
                        incident_filepath.parent.mkdir(parents=True, exist_ok=True)
                        incident_data = {
                            "product": product_name,
                            "product_url": product_url,
                            "summary": summary,
                            "date": date,
                            "duration": duration,
                            "report": incident_report,
                        }
                        write_json(incident_data, incident_filepath)
                        self.data.append(incident_data)
                        incidents_list.append(
                            {
                                "product": product_name,
                                # "product_url": product_url,
                                "date": date,
                                "duration": duration,
                                "id": incident_id,
                                "summary": summary,
                                "url": incident_url,
                            }
                        )
                        time.sleep(0.5)

            df_incidents = pd.DataFrame(incidents_list)
            df_incidents.to_csv(self.summary_filepath, sep="\t", index=False)

        else:
            df_incidents = pd.read_csv(self.summary_filepath, sep="\t")
            print(f"Original data: {len(df_incidents)}")
            h1_headers = []
            h2_headers = []
            h12_headers = []
            diagnosis = []

            df_incidents_filtered = df_incidents.drop_duplicates("id").copy()
            affected_products = []
            for _, incident in df_incidents_filtered.iterrows():
                x = df_incidents[df_incidents["id"] == incident["id"]]
                affected_services = x["product"].tolist()
                affected_products.append(affected_services)
            df_incidents_filtered["product"] = affected_products
            df_incidents_filtered["num_products"] = df_incidents_filtered["product"].apply(lambda x: len(x))
            df_incidents_filtered = df_incidents_filtered[
                ["product", "num_products", "date", "duration", "id", "url", "summary"]
            ]
            print(f"Data after deduplication: {len(df_incidents_filtered)}")

            # for _, incident in tqdm(df_incidents_filtered.iterrows(), total=len(df_incidents_filtered)):
            for _, incident in df_incidents_filtered.iterrows():
                incident_id = incident["id"]
                affected_services = incident["product"]
                summary = incident["summary"]
                date = incident["date"]
                duration = incident["duration"]
                url = incident["url"]

                # if not incident_id == "ETJGhvY9Xaktw7tgi8dF":
                #     continue

                incident_html_filepath = self.out_dir / "html" / f"{incident_id}.html"
                if incident_html_filepath.exists():
                    # print(f"Reading {incident_html_filepath.name}")
                    with open(incident_html_filepath, "r", encoding="utf-8") as f:
                        incident_page_source = f.read()
                else:
                    print(f"Crawling {url}")
                    incident_driver = get_webdriver()
                    incident_driver.implicitly_wait(5)
                    incident_driver.get(url)
                    incident_page_source = incident_driver.page_source
                    incident_html_filepath.parent.mkdir(parents=True, exist_ok=True)
                    with open(incident_html_filepath, "w", encoding="utf-8") as f:
                        f.write(incident_page_source)
                    incident_driver.quit()

                incident_soup = BeautifulSoup(incident_page_source, "html.parser")
                incident_report = self._extract_incident_report(incident_soup, incident_id)
                incident_filepath = self.out_dir / "json" / f"{incident_id}.json"
                incident_filepath.parent.mkdir(parents=True, exist_ok=True)

                post_morterm = {k: v for k, v in incident_report.items() if k in self.known_keys}
                incident_data = {
                    "affected_services": affected_services,
                    "num_affected_services": len(affected_services),
                    "url": url,
                    "track_id": incident_id,
                    "date": date,
                    "duration": duration,
                    # "summary": summary,
                    "title": incident_report["title"],
                    **post_morterm,
                    "human_labels": {},
                    "start_timezone": incident_report["start_timezone"],
                    "start_time": incident_report["start_time"],
                    "end_timezone": incident_report["end_timezone"],
                    "end_time": incident_report["end_time"],
                    "html": incident_report["html"],
                }

                # TODO
                h1_headers.append(incident_report["h1_headers"])
                h2_headers.append(incident_report["h2_headers"])
                h12_headers.append(incident_report["h12_headers"])
                # diagnosis.append(incident_report["diagnosis"])

                # write_json(incident_data, incident_filepath)
                self.data.append(incident_data)

            write_json(self.data, self.out_dir / "all_data.json")

            # df_incidents_filtered["diagnosis"] = diagnosis
            df_incidents_filtered["h1_headers"] = h1_headers
            df_incidents_filtered["h2_headers"] = h2_headers
            df_incidents_filtered["h12_headers"] = [len(i) for i in h12_headers]
            df_incidents_filtered_with_keys = pd.DataFrame(self.data)
            df_incidents_filtered_with_keys = df_incidents_filtered_with_keys[
                df_incidents_filtered_with_keys["root cause"].notna()
            ].reset_index(drop=True)
            incidents_filtered_with_keys = [
                dict(zip(row.keys(), row.values)) for _, row in df_incidents_filtered_with_keys.iterrows()
            ]
            write_json(incidents_filtered_with_keys, self.out_dir / "all_data_filtered.json")
            print(f"incidents_filtered_with_keys: {len(incidents_filtered_with_keys)}")

            # Stats
            h1_uniq, h1_count = np.unique(self.h1, return_counts=True)
            h2_uniq, h2_count = np.unique(self.h2, return_counts=True)
            with pd.ExcelWriter(self.summary_filepath.parent / f"{self.summary_filepath.stem}_filtered.xlsx") as writer:
                df_incidents_filtered.to_excel(writer, sheet_name="main", index=False)
                x = df_incidents_filtered[df_incidents_filtered["id"].isin(self.incidents_with_root_cause)]
                ids_with_rootcause = x["id"]
                x.to_excel(writer, sheet_name="ids_with_rootcause", index=False)
                x = df_incidents_filtered[df_incidents_filtered["id"].isin(self.incidents_with_remediation)]
                ids_with_remediation = x["id"]
                print(
                    "Non-overlap ids between ids_with_rootcause and ids_with_remediation: "
                    f"{ids_with_rootcause[~ids_with_rootcause.isin(ids_with_remediation)].tolist()}"
                )
                x.to_excel(writer, sheet_name="ids_with_remediation", index=False)
                df_h1 = pd.DataFrame({"h1_headers": h1_uniq, "h1_count": h1_count})
                df_h2 = pd.DataFrame({"h2_headers": h2_uniq, "h2_count": h2_count})
                df_h12 = pd.concat([df_h1, df_h2], axis=1, ignore_index=True)
                df_h12.to_excel(writer, sheet_name="stats", index=False)

        print(f"Num mini report: {self.num_mini_report}")
        print(list(set(self.ids_report) & set(self.ids_mini_report)))


def main():
    parser = ArgumentParser()
    parser.add_argument("--platform", choices=["cloud", "workspace"], default="cloud")
    parser.add_argument("-o", "--out_dir", required=True)
    args = parser.parse_args()

    crawler = GoogleCrawler(args.platform, out_dir=args.out_dir, headless=True, init_driver=False)
    crawler.run()
    crawler.cleanup()


if __name__ == "__main__":
    main()
