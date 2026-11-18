# Custom Indian dataset

`custom_india.csv` holds job/internship posts targeting Indian students. It is used **only for evaluation**
(to test whether a model trained on EMSCAD generalises to Indian scam patterns) — never for training,
unless you deliberately move it into training later.

| column  | meaning                                                                 |
|---------|-------------------------------------------------------------------------|
| text    | full post text (title + description)                                    |
| company | company name as written in the post ('' if none)                       |
| email   | contact e-mail in the post ('' if none)                                 |
| salary  | salary / stipend exactly as written ('' if none)                        |
| source  | where it came from: `internshala`, `linkedin`, `whatsapp_forward`, `telegram`, `synthetic_example`, ... |
| label   | **1 = fake/scam**, **0 = real**                                         |

## ⚠️ The 40 starter rows are SYNTHETIC
All rows with `source = synthetic_example` were written by hand to illustrate common Indian scam
patterns (registration fees, "earn ₹50k/week from home", WhatsApp/Telegram-only contact, Gmail
instead of a company domain, no interview) and realistic legitimate posts. Company names, e-mails and
phone numbers are fictional. **Replace them with real, collected and verified posts** before reporting
results in your résumé or interviews.

## Adding samples
```bash
# interactive
python data/custom_india/add_sample.py

# one-liner
python data/custom_india/add_sample.py --text "Earn 5000 daily..." --email abc@gmail.com --source whatsapp_forward --label 1

# bulk import a CSV that follows template.csv
python data/custom_india/add_sample.py --from-csv my_collected_posts.csv

# remove all synthetic starter rows once you have real data
python data/custom_india/add_sample.py --drop-synthetic
```

Tips for collecting real data: screenshot scam messages from college WhatsApp groups (remove personal
numbers), report-and-save fake posts on job boards, and label legitimate posts from verified company
career pages. Aim for at least 200 posts with a realistic class balance.
