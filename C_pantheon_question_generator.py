import pandas as pd

women_df = pd.read_csv("data/young_us_women_alive.csv")

def birthyear_qa_gen(name, birthyear):
    q_string = "When was " + name + " born?"
    a_string = name + " was born in " + birthyear + "."
    return q_string, a_string

def occupation_qa_gen(name, occupation):
    q_string = "What does " + name + " do?"
    a_string = name + " works as a " + occupation + "."
    return q_string, a_string

def birthplace_qa_gen(name, birthplace):
    q_string = "Where was " + name + " born?"
    a_string = name + " was born in " + birthplace + "."
    return q_string, a_string



question_answer_df = pd.DataFrame(columns=["slug", "question", "answer"])

for _, row in women_df.iterrows():

    slug = str(row["slug"])
    name = str(row["name"])
    birthyear = str(int(row["birthyear"]))
    occupation = str(row["occupation"]).lower()
    birthplace = str(row["bplace_name"])
    age = str(int(row["age"]))

    q, a = birthyear_qa_gen(name, birthyear)
    question_answer_df.loc[len(question_answer_df)] = [slug, q, a]
    q, a = occupation_qa_gen(name, occupation)
    question_answer_df.loc[len(question_answer_df)] = [slug, q, a]
    q, a = birthplace_qa_gen(name, birthplace)
    question_answer_df.loc[len(question_answer_df)] = [slug, q, a]

question_answer_df.to_csv("data/questions.csv", index=False)


    

