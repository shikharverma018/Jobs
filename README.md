# Job Skill Extractor & Resume Matching Engine

An end-to-end NLP project for extracting skills from job descriptions, parsing resumes, and building a recommendation engine to match candidates with job-specific skill profiles.

---

## Step 1: Data Acquisition (Web Scraping)

The project began by building a high-quality dataset of job descriptions.  
A custom web scraping script was developed to collect postings from various job portals using the following job titles:

- Data Analyst
- Data Engineer
- Data Scientist
- Machine Learning Engineer
- Cloud Engineer
- Cybersecurity Analyst
- Full Stack Developer
- Software Developer
- DevOps Engineer
- Database Administrator

This process resulted in **4,429 raw job descriptions**, stored in JSON format.  
This dataset serves as the foundation for training the Named Entity Recognition (NER) model used for skill extraction.
