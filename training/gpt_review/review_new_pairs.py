import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

data_dir = Path(os.environ['NAMA_DATA'])


with open(data_dir/'openai_key.txt','r') as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)


review_dirs = sorted((data_dir/'training_data'/'processed'/'gpt_review'/'iterations').iterdir())

for review_dir in review_dirs:
    try:
        review_df = pd.read_csv(review_dir/'new_review_pairs.csv')
    except FileNotFoundError:
        print(f'No review file found in {review_dir}')
        continue

    if (review_dir/'gpt_reviewed.csv').exists():
        # print(f'GPT review already exists in {review_dir}')
        continue

    print(f'Reviewing {review_dir}')

# review_df = review_df.sample(5)

    reviewed = []
    for i,j in tqdm(review_df[['i','j']].values,desc='Reviewing pairs'):
        prompt = f'''Do the following names refer to the same organization or parts of the same organization?
String 1: "{i}"
String 2: "{j}"
First, explain the relationship between the orgs in 10 words or less.
Then rate the probability on a 1-5 scale where "5" means "same organization, person, or parent company" and "1" means "different organization/person".
No further explanation is required.
'''.strip()

        response = client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ],
            model='gpt-4-1106-preview',
            temperature=0,
            max_tokens=100,
        )

        answer = response.choices[0].message.content

        # print(f'{i}\n{j}\n{answer}\n------')

        reviewed.append((i,j,answer))

    reviewed_df = pd.DataFrame(reviewed,columns=['i','j','gpt_rating'])

    reviewed_df.to_csv(review_dir/'gpt_reviewed.csv',index=False)



# review_df = pd.DataFrame([
#     ('BLUECROSS BLUESHIELD OF FLORIDA','BlueCross BlueShield of South Carolina'),
#     ('Boeing','Boeing Commercial Airplanes'),
#     ('UNIV OF CALIFORNIA','UNIV OF CALIFORNIA AT SAN DIEGO'),
#     ('ROLLS-ROYCE NORTH AMERICA, INC', 'Rolls-Royce Canada Limited'),
#     ('Government Solutions LLC', 'Government Solutions Group'),
#     ('BARCLAYS US GFP INC.','BARCLAYS US GPF INC.'),
#     ('City of Fayette','City of Fayette, Georgia'),
#     ('City of Fayette','City of Fayette, AL'),
#     ('City of Fayette, Georgia','City of Fayette, Alabama'),
#     ('FRED HIESTAND','Histand, Fred'),
#     ('SABRE','Sabres Holding Corporation'),
#     ('CC Services','COUNTRY Financial'),
#     ],columns=['i','j'])




# for i,j in tqdm(review_df[['i','j']].values,desc='Reviewing pairs'):
#         prompt = f'''Do the following names refer to the same organization or parts of the same organization?
# String 1: "{i}"
# String 2: "{j}"
# First, explain the relationship between the orgs in 10 words or less.
# Then rate the probability on a 1-5 scale where "5" means "same organization/person" and "1" means "different organization/person".
# No further explanation is required.
# '''.strip()

#         response = client.chat.completions.create(
#             messages=[
#                 {
#                     'role': 'user',
#                     'content': prompt,
#                 }
#             ],
#             model='gpt-4-1106-preview',
#             temperature=0,
#             max_tokens=100,
#         )

#         answer = response.choices[0].message.content

#         print(f'{i}\n{j}\n{answer}\n------')



# for i,j in tqdm(review_df[['i','j']].values,desc='Reviewing pairs'):
#         prompt = f'''Do the following strings refer to the same organization or parts of the same organization?
# String 1: "{i}"
# String 2: "{j}"
# Rate the probability on a 1-5 scale where "5" means "same organization" and "1" means "different organization".
# Print only the number.
# '''.strip()

#         response = client.chat.completions.create(
#             messages=[
#                 {
#                     'role': 'user',
#                     'content': prompt,
#                 }
#             ],
#             model='gpt-3.5-turbo',
#             temperature=0,
#             max_tokens=1,
#         )

#         answer = response.choices[0].message.content

#         print(f'{i}\n{j}\n{answer}\n------')