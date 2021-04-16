from scipy.stats import binom
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
sns.set_theme()

X = K = 2 # successes
N = 3 # trials
P = 0.75 # probability

# binom.
b = binom.pmf(k=X, n=N, p=P)
# print(b)

def bernoulli(X, N, P):
    sum = 0.0
    while X <= N:
        sum += binom.pmf(k=X, n=N, p=P)
        X += 1
    return sum

def doctors_to_successes(N):
    if N % 2 == 0:
        return N/2
    else:
        return int(math.ceil(N/2))

def graph_for_c():
    jurySizes = [1, 3, 9, 27, 51, 99]
    competenceLevels = np.arange(0.50, 1.05, 0.05)
    competenceLevels = [round(x, 3) for x in competenceLevels]

    probabilities = {}
    for size in jurySizes:
        temp_dict = {}
        for comp in competenceLevels:
            X = doctors_to_successes(size)
            prob = bernoulli(X=doctors_to_successes(size), N=size, P=comp)
            print("N:", size, ", X:", X, ", P:", comp, ", prob:", prob)
            temp_dict[str(comp)] = prob
        probabilities[str(size)] = temp_dict

    print(probabilities)
    df = pd.DataFrame(probabilities)

    print(df.head())
    sns.lineplot(data=df)
    plt.title("Probability of correct decision for various jury sizes and competence levels")
    plt.xlabel("Competence level")
    plt.xticks(rotation=0)
    plt.ylabel("Probability of correct decision")
    plt.legend(loc='lower right', title="Jury sizes")
    plt.show()

def graph_for_d():
    dict = {}
    dict["radiologist"] = bernoulli(X=doctors_to_successes(1), N=1, P=0.85)
    dict["doctors"] = bernoulli(X=doctors_to_successes(3), N=3, P=0.75)
    dict["students"] = bernoulli(X=doctors_to_successes(31), N=31, P=0.6)
    print(dict)
    keys = list(dict.keys())
    vals = [float(dict[k]) for k in keys]
    sns.barplot(x=keys, y=vals)
    plt.title("Probability of correct decision for three groups")
    plt.ylabel("Probability of correct decision")
    plt.ylim(0.8, 0.9)
    plt.show()

def graph_for_d2():
    jurySizes = np.arange(1, 31, 2)
    competenceLevel = 0.6

    probabilities = {}
    for size in jurySizes:
        temp_dict = {}
        X = doctors_to_successes(size)
        prob = bernoulli(X=doctors_to_successes(size), N=size, P=competenceLevel)
        print("N:", size, ", X:", X, ", P:", competenceLevel, ", prob:", prob)
        temp_dict[str(competenceLevel)] = prob
        probabilities[str(size)] = temp_dict

    print(probabilities)
    df = pd.DataFrame(probabilities).T

    print(df.head())
    sns.lineplot(data=df)
    plt.title("Probability of correct decision for varying number of students")
    plt.xlabel("Number of students")
    plt.xticks(rotation=0)
    plt.ylabel("Probability of correct decision")
    plt.legend(loc='lower right', title="Competence level")
    plt.show()

def graph_for_e():
    jurySizes = np.arange(1, 501, 2)
    competenceLevel = 0.6

    probabilities = []
    for size in jurySizes:
        X = doctors_to_successes(size)
        prob = bernoulli(X=doctors_to_successes(size), N=size, P=competenceLevel)
        print("N:", size, ", X:", X, ", P:", competenceLevel, ", prob:", prob)
        probabilities.append(prob)

    sns.lineplot(x=jurySizes, y=probabilities)
    plt.title("Probability of correct decision for varying number of students")
    plt.xlabel("Number of students")
    plt.ylabel("Probability of correct decision")
    plt.show()

# graph_for_c()
graph_for_d()
# graph_for_d2()
# graph_for_e()