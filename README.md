# Capstone Project
## Files

* ```data/SGJobData.csv.xz```, this file is the CSV file data source for the project, except compressed so that GitHub can accept it.
    * processed data are saved as ```'*.parquet'``` files, and prefixed with 'cleaned', 3 files:
        * pre-exploded categories
        * post-exploded categories
        * pre-explode categories with skills
* ```notebooks/*.ipynb```, jupyter notebook for testing out graphs and commands etc.
    * ```-eda.ipynb```, file that generated the cleaned parquet dfs
    * ```-ed-ml.ipynb```, can ignore, first trial for generating skillslist
    * ```visualplayground.ipynb```, code for the visual to test out before putting in app.py
* ```app.py```, main source code for streamlit application hosting
    * run this in streamlit

## Data: data/cleaned-sgjobdata-exploded.parquet
this is a list dataframe of job postings.
 0   job_id                  string     : unique identifier
 1   title                   string     : job title (dirty and messy)
 2   company                 string     : company name
 3   min_exp                 Int64      : minimum experience required for this job
 4   positionlevels          string     : position level of the job
 5   num_applications        Int64      : number of applications for this job
 6   num_views               Int64      : number of views for this job
 7   num_vacancies           Int64      : number of vacancies for this 1 job posting
 8   average_salary          Float64    : salary expected for this job
 9   average_salary_cleaned  Float64    : salary expected for this job, cleaned using windsorisation
 10  category                object     : sector category, in JSON format containing id, and category (name)
 11  jobtitle_cleaned        string     : cleaned job title 
 12  skill                   string     : skills needed for this particular job_id

## How to set up environment
uv is a Python environment manager. To set up the environment, run the following commands after you have installed uv:

1. ```uv venv .venv``` : this will setup the virtual environment
2. ```uv venv .venv --python 3.12``` : this sets up the venv with python 3.12
3. ```source .venv/bin/activate``` : this activates the venv
4. ```uv sync``` : this needs the uv.lock file, it installs everything needed for this environment

## How to Lock env and dependencies? [for maintainer only, not for end users]
1. create a pyproject.toml file
2. run: ```uv lock```

## How to Run Streamlit app:
1. navigate to project folder
2. run command ```streamlit run app.py```

## Data Cleaning Notes // Issues
1. NULLs dropped via Job ID
2. Drop unused columns and rename for clarity
3. Capping average_salary column using Log-IQR concept, this caps upper bound at 19783.0, lower bound at 1110.0
    * problem with this is that data is already flawed, small numbers indicate user behaviour where they choose not to fill in salary, large numbers can be for the same reason or mistaking the field for 'annual' salary.
4. matching of job title posted to job titles from SkillsFuture list of jobs and skills
    * not a fool proof 100% match, for e.g. "Driver" is matched to "Engine Driver" which is wrong, the correct match should be "Transport Operator", the matching process uses sentence transformer to get closest possible match however cultural semantics are lost in this 'translation'.
    * first try was abandoned, where an LLM is used to perform one-shot inference, the data is minimally cleaned and then combined with company name to provide even more context in the hopes of the llm being able to generate a better list of 'top skills', however, the generated list of skills follow no taxonomy so there is no standard and no consistetncy for proper analysis. using a governement approved taxonomy is better sense.

## Links
* [Skills Framework Dataset](https://jobsandskills.skillsfuture.gov.sg/frameworks/skills-frameworks)

## Git & GitHub
* ```eval "$(ssh-agent -s)" ```
* ```ssh-add ~/.ssh/id_ed25519```
* ```ssh-add -l  ```
* ```ssh -T git@github.com```