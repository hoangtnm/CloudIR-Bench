import calendar
import re
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
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


class GitHubCrawler(BaseCrawler):
    def __init__(self, url, headless, out_dir, implicit_wait=10, init_driver: bool = True):
        super().__init__(url, headless, implicit_wait, init_driver)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _get_incident_ids(self, driver):
        data = []

        back_btn = find_element(driver, By.CSS_SELECTOR, "div.pagination-container > div.pagination > a.previous-page")
        while not "disabled" in back_btn.get_attribute("class"):
            time.sleep(5)

            current = find_element(
                driver, By.CSS_SELECTOR, "div.pagination-container > div.pagination > span.current"
            ).text.strip()
            print(current, flush=True)

            for expand_btn in find_elements(driver, By.CSS_SELECTOR, "div.expand-incidents"):
                expand_btn.click()
            time.sleep(0.2)

            soup = get_soup_from_driver(driver)
            for incident in soup.select("div.incident-data.incident-container"):
                month_title = incident.parent.parent.parent.previous_sibling
                month_str = month_title.contents[0].text.strip()
                month = list(calendar.month_name).index(month_str)
                year = month_title.find("var").text.strip()
                secondary = incident.select_one("div.secondary")
                date = secondary.select_one("var[data-var=date]").text.strip()
                duration = secondary.select("var[data-var=time]")
                duration = " - ".join([i.text.strip() for i in duration])

                title_elm = incident.select_one("a.incident-title")
                impact = re.findall("impact-\w+", "|".join(title_elm.get_attribute_list("class")))[0].split("-")[1]
                incident_message = incident.select_one("div.message").text.strip()
                url = title_elm["href"].strip()
                data.append(
                    {
                        "date": f"{month}/{date}/{year}",
                        # "duration": duration,
                        "track_id": Path(url).name,
                        "url": url,
                        "impact": impact,
                        "title": title_elm.text.strip(),
                        "summary": incident_message,
                    }
                )

            back_btn.click()
        return data

    def _extract_incident_report(self, soup, incident_id):
        components_affected = soup.select_one("div.components-affected")
        if components_affected:
            components_affected.decompose()

        page_title = soup.select_one("div.page-title")
        incident_name = page_title.select_one("div.incident-name")
        sub_header = page_title.select_one("div.subheader").text.strip()
        sub_header = " ".join([x.strip() for x in sub_header.split()])
        data = {
            "title": incident_name.text.strip(),
            "sub_title": sub_header,
            "severity": re.findall("impact-\w+", "|".join(incident_name.get_attribute_list("class")))[0].split("-")[1],
            "updates": [],
            "raw_text": "",
            "human_labels": {},
        }

        updates_ele = soup.select("div.incident-updates-container > div.update-row")
        updates_ele = updates_ele[::-1]
        for idx, update in enumerate(updates_ele):
            update_status = update.select_one("div.update-title").text.strip().lower()
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
            # data["raw_text"] += f"[{update_status}] {update_time_str}: {update_message}\n\n"
            message_with_status = f"[{update_status}] {update_time_str}: {update_message}\n\n"
            message_without_status = f"{update_time_str}: {update_message}\n\n"
            if idx in [0, len(updates_ele) - 2, len(updates_ele) - 1]:
                data["raw_text"] += message_with_status
            else:
                data["raw_text"] += message_without_status

        updates = data["updates"]
        start_tz, end_tz = updates[0]["timezone"], updates[-1]["timezone"]
        start_time, end_time = updates[0]["timestamp"], updates[-1]["timestamp"]
        data["html"] = data["raw_text"].strip()
        data.pop("raw_text")
        data.update(
            {
                "status": data["updates"][-1]["status"],
                # "raw_text": data["raw_text"].strip(),
                # "html": str(soup.select_one("div.incident-updates-container")),
                "start_timezone": start_tz,
                "start_time": start_time,
                "end_timezone": end_tz,
                "end_time": end_time,
            }
        )
        return data

    def run(self) -> None:
        summary_filepath = self.out_dir / "github_summary.csv"
        if summary_filepath.exists():
            self.data = pd.read_csv(summary_filepath, sep="\t")
        else:
            self._init_driver(self.headless, self.implicit_wait, self.url)
            self.data = pd.DataFrame(self._get_incident_ids(self.driver))
            self.data.to_csv(summary_filepath, sep="\t", index=False)

        incidents_data = []
        incidents_status = []
        num_words_last_update = []
        num_words = []
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            incident_url = row["url"]
            incident_id = Path(row["url"]).name

            # print(f"Processing {incident_id}...")
            incident_html_filepath = self.out_dir / "html" / f"{incident_id}.html"
            incident_page_source = self.cache_html(incident_url, incident_html_filepath)
            incident_soup = BeautifulSoup(incident_page_source, "html.parser")
            incident_report = self._extract_incident_report(incident_soup, incident_id)
            incident_filepath = self.out_dir / "json" / f"{incident_id}.json"
            incident_filepath.parent.mkdir(parents=True, exist_ok=True)
            # assert row["impact"] == incident_report["impact"], f"{incident_id}"
            incidents_data.append(
                {
                    # "date": incident_report["updates"][-1]["update_time"],
                    "date": row["date"],
                    "track_id": incident_id,
                    "url": incident_url,
                    **incident_report,
                }
            )
            incidents_status.append(incident_report["status"])
            num_words.append(len(" ".join([i["message"] for i in incident_report["updates"]]).split(" ")))
            num_words_last_update.append(len(incident_report["updates"][0]["message"].split(" ")))
            # write_json(incident_report, incident_filepath)
        write_json(incidents_data, self.out_dir / "all_data.json")

        print(f"Before deduplication: {self.data.shape}")
        df_filtered = self.data.drop_duplicates(["date", "title", "summary"]).copy()
        print(f"After deduplication: {df_filtered.shape}")

        with open("github_whitelist.txt", "r") as f:
            ids_whitelist = f.readlines()
            ids_whitelist = [i.strip() for i in ids_whitelist]
        df_filtered = df_filtered[df_filtered["track_id"].isin(ids_whitelist)]
        print(f"After whitelist: {df_filtered.shape}")
        incidents_filtered = [x for x in incidents_data if x["track_id"] in ids_whitelist]
        write_json(incidents_filtered, self.out_dir / "all_data_filtered.json")
        print(df_filtered.shape)

        # Overwrites summary with incidents_status
        self.data["status"] = incidents_status
        self.data["num_words_last_update"] = num_words_last_update
        self.data["num_words"] = num_words
        ax = self.data["num_words"].hist(bins=20)
        for name in ["num_words", "num_words_last_update"]:
            plt.figure()
            ax = self.data[name].hist(bins=30)
            plt.xlabel("num_words")
            plt.ylabel("num_incidents")
            plt.savefig(self.out_dir / f"{name}.png")
            plt.close()
        self.data.to_csv(f"{summary_filepath.stem}_dedup.csv", sep="\t", index=False)
        x = self.data.drop_duplicates("track_id").copy()
        print(len(self.data))
        print(len(x))


def main():
    parser = ArgumentParser()
    parser.add_argument("--url", default="https://www.githubstatus.com/history")
    parser.add_argument("-o", "--out_dir", required=True)
    args = parser.parse_args()

    crawler = GitHubCrawler(args.url, out_dir=args.out_dir, headless=True, init_driver=False)
    crawler.run()
    crawler.cleanup()


if __name__ == "__main__":
    main()
