
# Chapter 12: Weasel Program
# Evolutionary Algorithms Using Mutation and Selection

import random
import string
from pathlib import Path
import matplotlib

# Save graphs without opening windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Save files in the same folder as Weasel.py
folder = Path(__file__).resolve().parent

# STEP 1: DEFINE TARGET PHRASE
target = "METHINKS IT IS LIKE A WEASEL"
characters = string.ascii_uppercase + " "


# STEP 2: FITNESS FUNCTION
def fitness(candidate):
    score = 0

    for a, b in zip(candidate, target):
        if a == b:
            score += 1

    return score


print("--- FITNESS FUNCTION TEST ---")
print("Target phrase:", target)
print("Target length:", len(target))
print("Perfect match score:", fitness(target))
print("Random phrase score:", fitness("ABCDEFGHIJKLMNOPQRSTUVWXYZAB"))


# STEP 3: MUTATION FUNCTION
def mutate(parent):
    new_string = ""

    for char in parent:
        # Each character has a 5% chance of mutation
        if random.random() < 0.05:
            new_string += random.choice(characters)
        else:
            new_string += char

    return new_string


print("\n--- MUTATION TEST ---")

original = target
mutated = mutate(original)

print("Original:", original)
print("Mutated: ", mutated)
print("Fitness Score:", fitness(mutated))


# STEP 4: EVOLUTION LOOP

# Generate a completely random starting phrase
best_parent = "".join(
    random.choice(characters) for _ in range(len(target))
)

best_score = fitness(best_parent)
generation = 0

# Store fitness scores and phrases for every generation
fitness_history = [best_score]
phrase_history = [best_parent]

print("\n--- EVOLUTION PROCESS ---")
print(
    f"Generation {generation}: "
    f"Score {best_score}/{len(target)} | {best_parent}"
)

# Continue until the target phrase is reached
while best_score < len(target):

    generation += 1

    # Create 100 mutated offspring
    offspring = [mutate(best_parent) for _ in range(100)]

    # Select the offspring with the highest fitness
    best_offspring = max(offspring, key=fitness)
    offspring_score = fitness(best_offspring)

    # Replace the parent only if the offspring is better
    if offspring_score > best_score:
        best_parent = best_offspring
        best_score = offspring_score

    # Record the best score and phrase
    fitness_history.append(best_score)
    phrase_history.append(best_parent)

    # Print the current generation
    print(
        f"Generation {generation}: "
        f"Score {best_score}/{len(target)} | {best_parent}"
    )


# STEP 5: FINAL RESULTS
print("\n--- EVOLUTION COMPLETED ---")
print("Final Phrase:", best_parent)
print("Final Fitness Score:", best_score)
print("Total Generations:", generation)


# STEP 6: CREATE AND SAVE FITNESS GRAPH
plt.figure(figsize=(10, 6))

plt.plot(
    range(len(fitness_history)),
    fitness_history,
    color="purple",
    linewidth=2,
    label="Best Fitness Score"
)

plt.axhline(
    y=len(target),
    color="green",
    linestyle="--",
    label="Perfect Fitness (28)"
)

plt.xlabel("Generation")
plt.ylabel("Fitness Score")
plt.title("Weasel Program: Evolution of Fitness")
plt.ylim(0, 30)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save graph without displaying it
graph_path = folder / "Weasel_fitness_progress.png"
plt.savefig(graph_path, dpi=300)
plt.close()

print("\nFitness graph saved successfully!")
print("Location:", graph_path)


# STEP 7: SAVE ALL GENERATIONS TO A TEXT FILE
text_path = folder / "Weasel_generations.txt"

with open(text_path, "w", encoding="utf-8") as file:
    file.write("WEASEL PROGRAM - EVOLUTION HISTORY\n")
    file.write(f"Target: {target}\n\n")

    for gen, (score, phrase) in enumerate(
        zip(fitness_history, phrase_history)
    ):
        file.write(
            f"Generation {gen}: "
            f"Score {score}/{len(target)} | {phrase}\n"
        )

    file.write("\n--- FINAL RESULTS ---\n")
    file.write(f"Final Phrase: {best_parent}\n")
    file.write(f"Final Fitness: {best_score}/{len(target)}\n")
    file.write(f"Total Generations: {generation}\n")

print("\nGeneration history saved successfully!")
print("Location:", text_path)

print("\nWeasel Program completed successfully!")
