import re
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from pprint import pprint
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from tqdm.auto import tqdm

from utils import (
    BaseCrawler,
    find_element,
    find_elements,
    flatten_chain,
    get_soup_from_driver,
    get_soup_from_url,
    get_webdriver,
    write_json,
)


class AtlassianCrawler(BaseCrawler):
    def __init__(self, url, headless, out_dir, implicit_wait=10, init_driver=True):
        super().__init__(url, headless, implicit_wait, init_driver)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _extract_incident_report(self, soup, url):
        page_title = soup.select_one("div.page-title")
        incident_name = page_title.select_one(".incident-name")
        sub_header = page_title.select_one(".subheader").text.strip()
        sub_header = " ".join([i.strip() for i in sub_header.split()])
        data = {
            "url": url,
            "title": incident_name.text.strip(),
            "sub_header": sub_header,
            "severity": re.findall("impact-\w+", "|".join(incident_name.get_attribute_list("class")))[0].split("-")[1],
            "updates": [],
            "raw_text": "",
            "human_labels": {},
        }

        updates_ele = soup.select("div.incident-updates-container > div.update-row")
        updates_ele = updates_ele[::-1]
        for idx, update in enumerate(updates_ele):
            update_status = update.select_one(".update-title").text.strip()
            update_message = update.select_one("div.update-container > div.update-body")
            for br in update_message.find_all("br"):
                br.replace_with(" ")
            update_message = self.remove_unicode_characters(update_message.text.strip())
            update_message = " ".join([x.strip() for x in update_message.split()])

            update_time_ele = update.select_one("div.update-container > div.update-timestamp")
            update_month = update_time_ele.select_one("span.ago").next_sibling.text.strip()
            update_date = update_time_ele.select_one("var[data-var=date]").text.strip()
            update_year = update_time_ele.select_one("var[data-var=year]").text.strip()
            update_time = update_time_ele.select_one("var[data-var=time]").text.strip()

            update_time_ele.find("span").decompose()
            x = update_time_ele.text.strip().split()
            timezone = x[-1]
            timestamp = " ".join(x[:-1])
            timestamp = datetime.strptime(timestamp, "Posted %b %d, %Y - %H:%M")
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M")
            update_time_str = f"{update_time} {timezone} on {update_date} {update_month} {update_year}"

            data["updates"].append(
                {
                    "status": update_status,
                    "message": update_message,
                    "update_time": update_time_str,
                    "timestamp": timestamp,
                    "timezone": timezone,
                }
            )
            message_with_status = f"[{update_status}] {update_time_str}: {update_message}\n\n"
            message_without_status = f"{update_time_str}: {update_message}\n\n"
            if idx in [0, len(updates_ele) - 2, len(updates_ele) - 1]:
                data["raw_text"] += message_with_status
            else:
                data["raw_text"] += message_without_status

        status = data["updates"][-1]["status"].lower()
        status = "resolved" if status in ["resolved", "postmortem"] else status
        data["status"] = status
        updates = data["updates"]
        start_tz, end_tz = updates[0]["timezone"], updates[-1]["timezone"]
        start_time, end_time = updates[0]["timestamp"], updates[-1]["timestamp"]
        data["html"] = data["raw_text"].strip()
        data.pop("raw_text")
        data.update(
            {
                "date": datetime.strptime(start_time, "%Y-%m-%d %H:%M").strftime("%Y-%m-%d"),
                "start_timezone": start_tz,
                "start_time": start_time,
                "end_timezone": end_tz,
                "end_time": end_time,
            }
        )

        messages_merged = "\n\n".join([x["message"] for x in data["updates"]])
        keywords_match = re.compile(
            r"\bSUMMARY\b|"
            r"\bBECAUSE\b|"
            r"\bDUE\ TO\b|"
            r"\bTRACED\ TO\b|"
            r"\bROOT\ CAUSE\b|"
            r"\bCAUS\w+\b|"
            r"\bLEAD\w*\ TO\b|"
            r"\bREMEDIA\w+\b|"
            r"\bMITIGAT\w+",
            re.IGNORECASE | re.X,
        )
        keywords_match = keywords_match.findall(messages_merged.lower())
        keywords_match = np.unique(keywords_match).tolist()
        data["keywords"] = keywords_match

        latest_update = updates_ele[-1]
        extract_headers = lambda tag: [el.text.lower().strip() for el in latest_update.select(tag)]
        data["h1_headers"] = extract_headers("h1")
        data["h2_headers"] = extract_headers("h2")
        data["h3_headers"] = extract_headers("h3")
        return data

    def crawl_reports_by_product(self, product, url):
        data = []
        driver = get_webdriver()
        driver.get(url)

        back_btn = find_element(driver, By.CSS_SELECTOR, "div.pagination-container > div > a.previous-page")
        while "disabled" not in back_btn.get_attribute("class"):
            time.sleep(2)
            current = find_element(
                driver, By.CSS_SELECTOR, "div.pagination-container > div.pagination > span.current"
            ).text.strip()

            for expand_btn in find_elements(driver, By.CSS_SELECTOR, "div.expand-incidents"):
                expand_btn.click()
            time.sleep(0.2)

            soup = get_soup_from_driver(driver)
            for incident in soup.select("div.incident-data.incident-container"):
                title_elm = incident.select_one("a.incident-title")
                impact = re.findall("impact-\w+", "|".join(title_elm.get_attribute_list("class")))[0].split("-")[1]
                incident_message = incident.select_one("div.message").text.strip()
                secondary = incident.select_one("div.secondary").text.strip()
                data.append(
                    {
                        "product": product,
                        "title": title_elm.text.strip(),
                        "duration": secondary,
                        "severity": impact,
                        "last_update": incident_message,
                        "url": title_elm["href"].strip(),
                    }
                )
            # break
            back_btn.click()

        driver.quit()
        return data

    def run(self) -> None:
        summary_filepath = self.out_dir / "atlassian_summary.csv"
        if summary_filepath.exists():
            self.data = pd.read_csv(summary_filepath, sep="\t")
        else:
            self._init_driver(self.headless, self.implicit_wait, self.url)
            soup = get_soup_from_driver(self.driver)
            products = soup.select("div.page-statuses-container a.page-status-container")
            summary = []
            for product in tqdm(products):
                product_name = product.select_one("div.status-info > div.product-name").text.strip()
                parsed_url = urlparse(product["href"])
                product_url = f"{parsed_url.scheme}://{parsed_url.netloc}/history"
                product_reports = self.crawl_reports_by_product(product_name, product_url)
                summary.append(product_reports)
            summary = flatten_chain(summary)
            self.data = pd.DataFrame(summary)
            self.data.to_csv(summary_filepath, sep="\t", index=False)

        incidents_data = []
        incidents_status = []
        h1_headers = []
        h2_headers = []
        h3_headers = []
        keywords = []
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            incident_url = row["url"]
            incident_id = Path(row["url"]).name

            incident_html_filepath = self.out_dir / "html" / f"{incident_id}.html"
            incident_page_source = self.cache_html(incident_url, incident_html_filepath)
            incident_soup = BeautifulSoup(incident_page_source, "html.parser")
            # print(f"Processing {incident_id}...")
            incident_report = self._extract_incident_report(incident_soup, incident_url)
            # incident_filepath = self.out_dir / "json" / f"{incident_id}.json"
            # incident_filepath.parent.mkdir(parents=True, exist_ok=True)

            incident_report["track_id"] = incident_id
            keywords.append(incident_report["keywords"])
            # incident_report.pop("keywords", None)
            incidents_data.append(incident_report)
            incidents_status.append(incident_report["status"])
            h1_headers.append(incident_report["h1_headers"])
            h2_headers.append(incident_report["h2_headers"])
            h3_headers.append(incident_report["h3_headers"])
            # write_json(incident_report, incident_filepath)
        write_json(incidents_data, self.out_dir / "all_data.json")

        # Overwrites summary with incidents_status
        self.data["status"] = incidents_status
        self.data["track_id"] = self.data["url"].apply(lambda x: Path(x).name)
        self.data["keywords"] = keywords
        self.data["updates"] = [report["html"] for report in incidents_data]
        self.data["h1_headers"] = h1_headers
        self.data["h2_headers"] = h2_headers
        self.data["h3_headers"] = h3_headers
        self.data.to_csv(summary_filepath, sep="\t", index=False)

        # def merge_df(source: pd.DataFrame, dest: pd.DataFrame):
        #     for _, row in source.iterrows():
        #         x = dest[dest[""]]

        df1 = self.data.drop_duplicates(["title", "duration", "last_update"]).copy()  # 3327
        df2 = self.data.drop_duplicates(["title", "duration"]).copy()  # 3316
        # x = pd.concat(
        #     [
        #         df1[["product", "title", "duration", "last_update", "url", "status"]],
        #         df2[["product", "title", "duration", "last_update", "url", "status"]],
        #     ],
        # ).drop_duplicates(keep=False)
        # x.to_csv("x.csv", index=False)
        df_filtered = self.data.drop_duplicates(["title", "duration", "last_update"]).copy()
        print(self.data.shape, df1.shape, df2.shape, df_filtered.shape)
        # df1.to_csv("df1.csv", index=False)
        # df2.to_csv("df2.csv", index=False)
        df_filtered.to_csv("df_filtered.csv", index=False)
        df_filtered["track_id"].to_csv("df_filtered_track_id.csv", index=False)

        incident_data_filtered = []
        with open("atlassian_whitelist.txt", "r") as fp:
            track_ids_whitelist = [i.strip() for i in fp.readlines()]
        # for incident_report in tqdm(incidents_data):
        #     if incident_report["track_id"] not in track_ids_whitelist:
        #         continue
        #     if not incident_report["keywords"]:
        #         continue
        #     if incident_report["severity"] == "maintenance":
        #         continue
        #     if incident_report["track_id"] not in df_filtered["track_id"].tolist():
        #         # print("....", incident_report["track_id"], incident_report["track_id"] in df_filtered["track_id"])
        #         continue
        #     incident_data_filtered.append(incident_report)
        incident_data_filtered = [i for i in incidents_data if i["track_id"] in track_ids_whitelist]
        print(f"incident_data_filtered: {len(incident_data_filtered)}")
        write_json(incident_data_filtered, self.out_dir / "all_data_filtered.json")

        self.data_postmortem = self.data[self.data["status"] == "Postmortem"]
        with pd.ExcelWriter(summary_filepath.parent / f"{summary_filepath.stem}_filtered.xlsx") as writer:
            self.data.to_excel(writer, sheet_name="main", index=False)
            self.data_postmortem.to_excel(writer, sheet_name="postmortem", index=False)
            self.data.drop_duplicates(["title", "duration", "last_update"]).copy().to_excel(
                writer, sheet_name="test", index=False
            )
            self.data_postmortem.drop_duplicates(["title", "duration", "last_update"]).copy().to_excel(
                writer, sheet_name="test_filtered", index=False
            )

            h1_uniq, h1_count = np.unique(flatten_chain(self.data_postmortem["h1_headers"]), return_counts=True)
            h2_uniq, h2_count = np.unique(flatten_chain(self.data_postmortem["h2_headers"]), return_counts=True)
            h3_uniq, h3_count = np.unique(flatten_chain(self.data_postmortem["h3_headers"]), return_counts=True)
            df = pd.concat(
                [
                    pd.DataFrame({"h1_headers": h1_uniq, "h1_count": h1_count}),
                    pd.DataFrame({"h2_headers": h2_uniq, "h2_count": h2_count}),
                    pd.DataFrame({"h3_headers": h3_uniq, "h3_count": h3_count}),
                ],
                axis=1,
                ignore_index=True,
            )
            df.to_excel(writer, sheet_name="stats", index=False)


def main():
    parser = ArgumentParser()
    parser.add_argument("--url", default="https://status.atlassian.com")
    parser.add_argument("-o", "--out_dir", required=True)
    args = parser.parse_args()

    crawler = AtlassianCrawler(args.url, out_dir=args.out_dir, headless=True, init_driver=False)
    crawler.run()
    crawler.cleanup()


if __name__ == "__main__":
    main()
