import pandas as pd
import random

other = ["UserID", "Weight (lbs)", "Height"]
allergens = ['Fish', 'Soybeans', 'Wheat', 'Gluten', 'Milk', 'Tree Nuts', 'Eggs', 'Sesame', 'Crustacean Shellfish']
dietary_restrictions = ['Gluten Friendly', 'Less Sodium', 'Smart Check', 'Vegan', 'Vegetarian', 'Contains Nuts']
dHalls = ["gelfenbien", "kosher", "north", "northwest", "McMahon", "putnam", "south", "whitney"]

def generateRandomUserDataset():

    L = other + allergens + dietary_restrictions + dHalls
    matrix = pd.DataFrame(columns=L)

    # For each user
    for i in range(1, 10):
        row = []        
        
        weight = random.randint(80, 300)
        feet = str(random.randint(4, 7))
        inches = str(random.randint(0, 12))
        height =  "{}'{}".format(feet, inches) 
        
        row.append(i) # userid
        row.append(weight) 
        row.append(height)


        for allergen in allergens:
            rand_num = random.randint(0, 1)
            row.append(rand_num)

        for diet_rest in dietary_restrictions:
            rand_num = random.randint(0, 1)
            row.append(rand_num)

        # User only prefers one dining hall
        rand_dHall = random.choice(dHalls)
        for dHall in dHalls:
            if dHall == rand_dHall:
                row.append(1)
            else:
                row.append(0)

        # Add each row to dataframe
        item = pd.DataFrame([row], columns = L)
        print(row)
        matrix = pd.concat([matrix, item])

    return matrix


generateRandomUserDataset()