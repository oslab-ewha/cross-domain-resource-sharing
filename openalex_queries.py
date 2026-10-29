"""Reproduces the documented literature search of Section 2.4 / Table A1.

Each query is matched against titles and abstracts in OpenAlex
(filter title_and_abstract.search), restricted to journal articles, conference
papers, and preprints (OpenAlex types article and preprint) published 2016-2026.
The search reported in the manuscript was run on 28 October 2026; later runs can
return slightly different counts because the index is updated continuously.

Run:  python openalex_queries.py            (writes records.csv)
"""
import csv, json, time, urllib.parse, urllib.request

QUERIES = {
 "Q1": '("volunteer computing" OR "volunteer edge") AND (edge OR GPU OR deadline OR "deep learning")',
 "Q2": '"edge federation" OR "fog federation" OR "cloudlet sharing" OR "edge resource sharing"',
 "Q3": '("GPU sharing" OR "idle GPUs") AND (campus OR volunteer OR decentralized)',
 "Q4": '("decentralized training" OR "decentralized inference") AND ("large language model" OR "foundation model" OR LLM)',
 "Q5": '("collaborative inference" OR "distributed inference") AND ("large language model" OR LLM) AND edge',
 "Q6": '("preemptible instances" OR "spot instances") AND (training OR serving) AND ("deep learning" OR LLM)',
 "Q7": '("task replication" OR "node churn") AND (edge OR vehicular OR volunteer)',
 "Q8": '(incentive OR auction OR contract) AND "resource sharing" AND (edge OR fog)',
 "Q9": '("trusted execution" OR "confidential computing") AND (GPU OR accelerator)',
 "Q10": '"geo-distributed" AND (GPU OR GPUs) AND (training OR inference)',
 "Q11": '"computing power network" OR "computing force network"',
 "Q12": '("parked vehicle" OR "vehicular fog" OR "vehicular cloud") AND (replication OR contract OR auction)',
 "Q13": '("heterogeneous GPUs" OR "consumer GPUs" OR "consumer-grade GPUs") AND ("large language model" OR LLM)',
 "Q14": '("volunteer resources" OR "idle resources" OR "contributed resources") AND (edge OR GPU)',
 "Q15": '("GPU" OR "GPUs") AND (preemptible OR reclaim OR revocable) AND training',
}
BASE = "https://api.openalex.org/works"

def fetch(url):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(url)

records = {}
for qid, q in QUERIES.items():
    page, n = 1, 0
    while True:
        flt = f"title_and_abstract.search:{q},publication_year:2016-2026,type:article|preprint"
        url = BASE + "?" + urllib.parse.urlencode({"filter": flt, "per-page": 50, "page": page,
                                                    "select": "id,doi,display_name,publication_year,type"})
        res = fetch(url)["results"]
        if not res:
            break
        for w in res:
            n += 1
            rec = records.setdefault(w["id"], {"title": w["display_name"], "year": w["publication_year"],
                                               "doi": w["doi"], "type": w["type"], "queries": []})
            if qid not in rec["queries"]:
                rec["queries"].append(qid)
        page += 1
        time.sleep(1)
    print(qid, n)
print("distinct records:", len(records))
with open("records.csv", "w", newline="", encoding="utf8") as f:
    wr = csv.writer(f)
    wr.writerow(["openalex_id", "year", "type", "doi", "queries", "title"])
    for k, r in records.items():
        wr.writerow([k, r["year"], r["type"], r["doi"], "+".join(r["queries"]), r["title"]])
