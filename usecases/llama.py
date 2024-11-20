import ollama
import subprocess
import time
import ast
entity_prompt = '''
            ## Instruction
            You are an expert in knowledge graph. Given a few examples, extract the "entity" from the last question. The answer must be extracted from the question without modifying. Please provide only the answer following the guidance and format in examples, without any additional information.

            ## Examples
            1. 
                - Question: 
                    "Who is the director of Good Will Hunting?"
                - Answer:
                    Good Will Hunting
            2. 
                - Question: 
                    "Who is the screenwriter of The Masked Gang: Cyprus?"
                - Answer: 
                    The Masked Gang: Cyprus
            3. 
                - Question: 
                    "Who is the director of Star Wars: Episode VI - Return of the Jedi?"
                - Answer: 
                    Star Wars: Episode VI - Return of the Jedi

            ## The Last Question
            "XXXXXXX"
'''

relation_prompt = '''
            ## Instruction
            You are an expert in knowledge graph. Given a few examples, extract the "predicate" from the last question. The answer must be extracted from the question without modifying. Please provide only the answer following the guidance and format in examples, without any additional information.

            ## Examples
            1. 
                - Question: 
                    "Who is the director of Good Will Hunting?"
                - Answer:
                    director
            2. 
                - Question: 
                    "Who is the screenwriter of The Masked Gang: Cyprus?"
                - Answer: 
                    screenwriter
            3. 
                - Question: 
                    "What is the MPAA film rating of Weathering with You?"
                - Answer: 
                    MPAA film rating

            ## The Last Question
            "XXXXXXX"
'''

classify_prompt = '''
    You are an AI trained to classify questions. Determine if the given question is a recommendation question. A recommendation question specifically asks for suggestions, such as recommending movies, books, or other items.
    Examples:
        Not a recommendation question:
            Who is the director of Star Wars: Episode VI - Return of the Jedi?
            Who directed the movie Apocalypse Now?
            What is the genre of Good Neighbors?
            Let me know what Sandra Bullock looks like.
            Show me a picture of Halle Berry.

        Recommendation question:
            Recommend movies similar to Apocalypse Now and Good Neighbors.
            Recommend movies like Star Wars: Episode VI - Return of the Jedi, Friday the 13th, and Apocalypse Now.
            Given that I like Hamlet, Othello, and The Beauty and the Beast, can you recommend some movies? 

    Task: Classify the following question. Determine if the given question is a recommendation question. Warning: If it is a recommendation question, output "Yes"; otherwise, output "No" Warning: Do not output anything other than one of these two options.

    Question:
    XXXXXXX
'''

entitylist_prompt = '''
            ## Instruction
            You are an expert in knowledge graph. Given a few examples, extract all entitys from the last question. There maybe more than one entity. The entitys must be extracted from the question without modifying. Please output only the entitys following the guidance and format in examples, without any additional information.

            ## Examples
            1. 
                - Question: 
                    "Who is the director of Good Will Hunting?"
                - Answer:
                    ["Good Will Hunting"]
            2. 
                - Question: 
                    "Recommend movies similar to Apocalypse Now and Good Neighbors."
                - Answer: 
                    ["Apocalypse Now", "Good Neighbors"]
            3. 
                - Question: 
                    "Given that I like Hamlet, Othello, and The Beauty and the Beast, can you recommend some movies?"
                - Answer: 
                    ["Hamlet", "Othello", "The Beauty and the Beast"]

            ## The Last Question
            "XXXXXXX"
'''

def entity_relation_extract(input):
    entity = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": entity_prompt.replace("XXXXXXX", input), 
            },
        ]
    )
    relation = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": relation_prompt.replace("XXXXXXX", input), 
            },
        ]
    )
    print("!entity:! ", entity["message"]["content"])
    print("!relation:! ", relation["message"]["content"])
    return entity["message"]["content"], relation["message"]["content"]

def classify_question(input):
    type = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": classify_prompt.replace("XXXXXXX", input), 
            },
        ]
    )
    return type["message"]["content"] 

def entitylist_extract(input):
    entitylist = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": entitylist_prompt.replace("XXXXXXX", input), 
            },
        ]
    )
    print("!entity_list:! ", entitylist["message"]["content"])
    return ast.literal_eval(entitylist["message"]["content"])

# def refine_wording(input):
#     prompt = '''
#     Please refine the following response to make it sound more personable, without adding any extra information. Do not change the content within parentheses.
#     Here is the response you need to modify:XXXXXXX
#     '''
#     ret = ollama.chat(
#         model="llama3.2",
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt.replace("XXXXXXX", input), 
#             },
#         ]
#     )
    
#     return ret["message"]["content"].replace("(","").replace(")","")# .replace("'","").replace('"','')
    


if __name__ == '__main__':
    # pip install ollama
    # ollama serve
    # ollama pull llama3.2
    command = ["ollama", "serve"]
    process = subprocess.Popen(command)
    time.sleep(10)
    
    # print(entity_relation_extract("What is the genre of Good Neighbors?"))
    # print(refine_wording("Apologies, but there is no corresponding answer in the database for your question."))
    # print(refine_wording("The factual answer is: (Richard Marquand), and the answers suggested by (embeddings) are: (Frank Oz,George Lucas,Lawrence Kasdan)."))
    # print(refine_wording("The factual answer is: (Gus Van Sant), and the answers suggested by (embeddings) are: (Harmony Korine,Ben Affleck,Gus Van Sant)."))
    # print(refine_wording("Apologies, but there is no corresponding answer in the database for your question."))
    # print(refine_wording("The factual answer is: ({}).".format("sutong")))
    # print(refine_wording("The answers suggested by (embeddings) are: ({}).".format("Tong Su")))
    print(classify_question("I just watched a movie called Naruto, can you recommend some similar movie?"))
    print(classify_question("I just watched a movie called Naruto, I like it. Is there any similar movie?"))
    print(classify_question("Recommend movies like Nightmare on Elm Street, Friday the 13th, and Halloween."))
    print(classify_question("Given that I like The Lion King, Pocahontas, and The Beauty and the Beast, can you recommend some movies?"))
    print(classify_question("When was 'The Godfather' released? "))
    print(classify_question("Who is the director of Star Wars: Episode VI - Return of the Jedi? "))
    print(classify_question("What is the MPAA film rating of Weathering with You? "))
    
    print(entitylist_extract("I just watched a movie called Naruto, can you recommend some similar movie?"))
    print(entitylist_extract("I just watched a movie called Naruto, I like it. Is there any similar movie?"))
    print(entitylist_extract("Recommend movies like Nightmare on Elm Street, Friday the 13th, and Halloween."))
    print(entitylist_extract("Given that I like The Lion King, Pocahontas, and The Beauty and the Beast, can you recommend some movies?"))
    print(entitylist_extract("When was 'The Godfather' released? "))
    print(entitylist_extract("Who is the director of Star Wars: Episode VI - Return of the Jedi? "))
    print(entitylist_extract("What is the MPAA film rating of Weathering with You? "))
    
    
    
    process.terminate()
    process.wait()
