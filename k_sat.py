from string import ascii_lowercase,ascii_uppercase
from random import sample
from itertools import combinations

m = int(input("Enter the number of clauses: "))
n = int(input("Enter number of variables: "))
k = int(input("Enter the number of variables in a clause: "))
N = int(input("Enter the number of such expression to generate: 
"))

def createProblem(m, k, n,N):
    positive_var = (list(ascii_lowercase))[0:n]
    negative_var = (list(ascii_uppercase))[0:n]
    variables = positive_var + negative_var
    problems = []
    all = list(combinations(variables, k))

    for i in range(N):
        s = sample(all, m)
        if s not in problems:
            problems.append(list(s))   
    return problems

problems = createProblem(m, k, n,N)

for i in range(len(problems)):
    print(problems[i])