## Populating Dataset

To populate our parquets with movie scripts and metadata, we used the [Movie Script Database](https://github.com/Aveek-Saha/Movie-Script-Database.git) repository to scrape, aggrgeate and clean this data. The process goes as follows:

```bash
git clone https://github.com/Aveek-Saha/Movie-Script-Database.git
cd Movie-Script-Database
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements
python get_scripts.py
```

After running for about (x) hours, this repo scraped 9 sites for a total of (x) movie scripts. We then have the following folder structure:
```
scripts
│
├── unprocessed // Scripts from sources
│   ├── source1
│   ├── source2
│   └── source3
│
├── temp // PDF files from sources
│   ├── source1
│   ├── source2
│   └── source3
│
├── metadata // Metadata files from sources/cleaned metadata
│   ├── source1.json
│   ├── source2.json
│   ├── source3.json
│   └── meta.json
│
├── filtered // Scripts with duplicates removed
│
└── parsed // Scripts parsed using the parser
    ├── dialogue
    ├── charinfo
    └── tagged
```

We must drop unneccessary data and aggrgeate this into a parquet, which has done by [this](fede/data/process-scraping-data.py) python script. 