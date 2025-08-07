import math
import os
import re
from datetime import datetime
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Union

import numpy as np
import pytz
import streamlit as st
from utils import read_json, write_json

st.set_page_config(layout="wide")


# @st.cache_data
def fetch_data(filepath: Union[Path, str]):
    data = read_json(filepath)
    return data


def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.cache_data.clear()


# @st.cache_data
# def get_metrics():
#     return {
#         "bleu": evaluate.load("bleu", smooth=True),
#         "rouge": evaluate.load("rouge"),
#         "bertscore": evaluate.load("bertscore"),
#         "meteor": evaluate.load("meteor"),
#     }


def app():
    # LCS_RESULTS = read_json("lcs_results.json")
    DATASETS = {
        "azure": "azure/all_data.json",
        "google_cloud": "google_cloud/all_data_filtered.json",
        "google_workspace": "google_workspace/all_data_filtered.json",
        "github": "github/all_data_filtered.json",
        "atlassian": "atlassian/all_data_filtered.json",
    }
    # DATA_DIR = Path("../data10samples")
    # DATASETS = {
    #     k: DATA_DIR / k / "annotations" / "final.json"
    #     for k in ["azure", "google_cloud", "google_workspace", "github", "atlassian"]
    # }

    st.subheader("Labeler (who is labeling the data)")
    labeler_names = [
        "hoang",
        "hoang-raw",
        "huong",
        "huong-raw",
        "tri",
        "other",
        "final",
    ]
    labeler = st.radio(
        "Labeler (who is labeling the data)",
        labeler_names,
        index=None,
        label_visibility="collapsed",
    )

    with st.sidebar:
        dataset = st.radio("Dataset", DATASETS.keys())
        dataset_filepath = Path(DATASETS[dataset])
        data = fetch_data(dataset_filepath)
        label_filepath = dataset_filepath.resolve().parent / f"{labeler}.json"
        HUMAN_LABELS = read_json(label_filepath) if label_filepath.resolve().exists() else {}

        track_ids = [i["track_id"] for i in data]
        track_id_captions = []
        for track_id in track_ids:
            status = HUMAN_LABELS.get(track_id, {})
            is_confused = status.get("is_confused")
            is_saved = status.get("is_saved", False)
            is_completed = status.get("is_completed", False)
            completed_status = "completed" if is_completed else "not_completed"
            saved_status = "saved" if is_saved else "not_saved"
            caption = f"{completed_status} & {saved_status}"
            if not status.get("is_included"):
                caption += " (not included)"
            if is_confused:
                caption = f":rainbow[{caption}]"
            track_id_captions.append(caption)

        if dataset == "azure":
            incident_col_spec = [0.3, 0.75]
        elif dataset == "github":
            incident_col_spec = [0.33, 0.7]
        elif dataset == "atlassian":
            incident_col_spec = [0.33, 0.67]
        else:
            incident_col_spec = 2
        col1, col2 = st.columns(incident_col_spec, gap="small")
        track_id_radio = col1.radio(f"Incident IDs: {len(track_ids)}", track_ids)
        col2.radio("Status", track_id_captions, index=None, disabled=True)
        current_id_idx = track_ids.index(track_id_radio)
        incident_data = [i for i in data if i["track_id"] == track_id_radio][0]
        incident_data = {k: v for k, v in incident_data.items() if not (isinstance(v, float) and np.isnan(v))}
        incident_data.update(
            {
                k: incident_data.get(k, "N/A")
                for k in [
                    "summary",
                    "impact",
                    "root cause",
                    "mitigation",
                    "long-term mitigation",
                    "diagnosis",
                    "workaround",
                    "updates",
                    "severity",
                    "status",
                ]
            }
        )
        WORKING_LABEL = HUMAN_LABELS.get(track_id_radio, {})
        # TODO
        # GPT_DATA = read_json("gpt-3.5-turbo-0125_parsed.json")[dataset][track_id_radio]
        # GPT_PREDICTIONS = GPT_DATA["predictions"]
        # GPT_GROUND_TRUTH = read_json(f"../data/{dataset}/annotations_summarization/gpt-3.5-turbo-0125_results.json")

    st.subheader(f"Title: {incident_data['title']}")
    st.write(f"Date: {incident_data['date']}\n\n" + f"Track_ID: {incident_data['track_id']}")
    if incident_data.get("url"):
        st.subheader(f"URL: {incident_data['url']}")
    WORKING_LABEL["title"] = incident_data["title"]

    include_keys = [
        "html",
        "raw_text",
        "summary",
        "impact",
        "root cause",
        "mitigation",
        "long-term mitigation",
        "diagnosis",
        "workaround",
        "updates",
        "severity",
        "status",
    ]
    incidenet_data_lite = {k: incident_data[k] for k in include_keys if k in incident_data}
    WORKING_LABEL.update(
        {
            k: WORKING_LABEL.get(k, incident_data.get(k))
            for k in [
                "severity",
                "status",
                "start_timezone",
                "start_time",
                "end_timezone",
                "end_time",
            ]
        }
    )

    # def widget_callback(labeler, incident_key, st_key) -> None:
    #     incidenet_data_lite["human_labels"][labeler][incident_key] = st.session_state[st_key]
    #     WORKING_LABEL[incident_key] = st.session_state[st_key]
    #     print(WORKING_LABEL)

    for k, v in incidenet_data_lite.items():
        v = v if v else ""

        if k == "html":
            st.subheader("Raw text")
            with st.container(border=True):
                if dataset.startswith(("github", "google")):
                    st.text(v)
                else:
                    st.markdown(v, unsafe_allow_html=True)
            continue

        st.subheader(k.capitalize())

        if k == "status":
            placeholder = "resolved/updating/investigating"
        elif k == "severity":
            placeholder = "critical/major/minor/none"
        else:
            placeholder = "N/A"

        text_height = 68 if k in ["severity", "status"] else 500
        st.write(f"Placeholder: {placeholder}")

        # TODO: Display GPT-3.5-turbo-0125 predictions
        if k in ["summary", "root cause", "mitigation"]:
            col1, col2 = st.columns(2)
            WORKING_LABEL[k] = st.text_area(
                k,
                value=WORKING_LABEL.get(k),
                placeholder=placeholder,
                height=text_height,
                label_visibility="collapsed",
            )
            # if k == "summary":
            #     col2_text = GPT_GROUND_TRUTH.get(track_id_radio, {}).get(k, "")
            # else:
            #     col2_text = LCS_RESULTS[dataset].get(track_id_radio, {}).get(k, "")
            # col2.text_area(
            #     f"{k} (GPT-3.5-turbo-0125)",
            #     value=col2_text,
            #     height=text_height,
            #     label_visibility="collapsed",
            #     disabled=True,
            # )
        else:
            WORKING_LABEL[k] = st.text_area(
                k,
                value=WORKING_LABEL.get(k),
                placeholder=placeholder,
                height=text_height,
                label_visibility="collapsed",
            )
        if k == "severity" and WORKING_LABEL[k] not in ["critical", "major", "minor", "none"]:
            st.write(f":red[Severity: {WORKING_LABEL[k]} is not valid]")
        if k == "status" and WORKING_LABEL[k] not in ["resolved", "updating", "investigating"]:
            st.write(f":red[Status: {WORKING_LABEL[k]} is not valid]")
        if not WORKING_LABEL[k]:
            WORKING_LABEL[k] = "N/A"
        # WORKING_LABEL[k] = " ".join([x.strip() for x in WORKING_LABEL[k].split()])
        # WORKING_LABEL[k] = re.sub(r"\s+", " ", WORKING_LABEL[k])
        WORKING_LABEL[k] = "\n".join([x.strip() for x in WORKING_LABEL[k].split("\n")])
        WORKING_LABEL[k] = WORKING_LABEL[k].strip()

    st.subheader("Timestamp")
    st.write("These info is pre-filled for Google, GitHub and Atlassian datasets")
    st.write("Format: YYYY-MM-DD HH:mm")
    col1, col2 = st.columns(2)
    start_timezone = col1.text_input("Start timezone", value=WORKING_LABEL.get("start_timezone", "UTC"))
    start_time = col2.text_input("Start time", value=WORKING_LABEL.get("start_time"))
    col1, col2 = st.columns(2)
    end_timezone = col1.text_input("End timezone", value=WORKING_LABEL.get("end_timezone", "UTC"))
    end_time = col2.text_input("End time", value=WORKING_LABEL.get("end_time"))

    def try_parse_datetime(x, tag):
        try:
            x = datetime.strptime(x, "%Y-%m-%d %H:%M")
            st.write(f"Parsed {tag} successfully")
        except Exception:
            st.write(f"Cannot parse {tag}")
            return False
        return True

    if start_time:
        start_parse_res = try_parse_datetime(start_time, "start")
    if end_time:
        end_parse_res = try_parse_datetime(end_time, "end")

    WORKING_LABEL["is_confused"] = st.toggle(
        "I'm not too sure about the label",
        value=WORKING_LABEL.get("is_confused", False),
    )
    WORKING_LABEL["is_included"] = st.toggle(
        "Include this report into the dataset",
        value=WORKING_LABEL.get("is_included", False),
    )
    WORKING_LABEL["is_completed"] = st.toggle(
        "Labeling completed",
        value=WORKING_LABEL.get("is_completed", False),
    )
    if st.button("Save all"):
        print(f"Saving all {track_id_radio}: {label_filepath}")
        WORKING_LABEL["is_saved"] = True
        WORKING_LABEL.update(
            {
                "is_saved": True,
                "start_timezone": start_timezone,
                "start_time": start_time,
                "end_timezone": end_timezone,
                "end_time": end_time,
            }
        )
        WORKING_LABEL["last_save_time"] = datetime.now(pytz.timezone("UTC")).strftime("%Y-%m-%d %H:%M")
        # WORKING_LABEL["last_save_time"] = datetime.now(pytz.timezone("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d %H:%M")
        HUMAN_LABELS[track_id_radio] = WORKING_LABEL
        if not (start_time and end_time):
            st.write("Cannot save as timestamp is empty")
        elif not (start_timezone and end_timezone):
            st.write("Cannot save as timezone is empty")
        else:
            if start_parse_res and end_parse_res:
                write_json(HUMAN_LABELS, label_filepath)
                st.rerun()
            else:
                st.write("Cannot save as timestamp is not correct")


if __name__ == "__main__":
    app()
