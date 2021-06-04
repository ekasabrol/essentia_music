import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'google-cloud-storage'])

from google.cloud import bigquery
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(input('Input path to json file with your credentials: '))

bigquery_client = bigquery.Client(project='switch-vc')

query ="""
SELECT  
   A.uuid, A.company_name, A.category_list, A.category_group_list
FROM
  `switch-vc.founder_crunchbase.organizations` AS A
INNER JOIN
  `switch-vc.analysis.cb_orgs_final_selected_v2` AS B
ON
  A.uuid = B.cb_uuid
WHERE 
  regexp_contains(A.category_list, r"^.*health|pharma|thera|bio|medic|life science+")
AND NOT 
  ( NOT regexp_contains(A.category_group_list, r"^.*biotechnology|advertising|financial|thera|clothing|pharma|consult|admin|lifestyle|sustainability|travel|food|professional|transport|media|real|payment|publishing|media|sales|commerce|government|gaming|messaging|education|sport|design|agriculture+")
    and 
  NOT regexp_contains(A.category_list, r"^.*hsport|tutoring|cosmetic|sms|diabetes|nutrition|dental|medicine|charity|family|wellness|hospital|consumption|women|agency|non profit|enterprise|employment|billing|clinical|trials|navigation|dietary|hospitality|elder|environ|beauty|advert|agriculture|energy|real estate|environ|news|developer|child care|consulting|rehab|nurs|government|food+"))  
"""


query_job = bigquery_client.query(query)
df = query_job.to_dataframe()
print(len(df))
print(df.head())
