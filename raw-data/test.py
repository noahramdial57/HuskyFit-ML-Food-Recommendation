import pandas as pd

meals = ['Food Item','Breakfast', 'Lunch', 'Dinner']
allergens = ['Fish', 'Soybeans', 'Wheat', 'Gluten', 'Milk', 'Tree Nuts', 'Eggs', 'Sesame', 'Crustacean Shellfish']
dietary_restrictions = ['Gluten Friendly', 'Less Sodium', 'Smart Check', 'Vegan', 'Vegetarian', 'Contains Nuts']

def preProcessData(dining_hall):

    df = pd.read_csv("./raw-data/{}.csv".format(dining_hall), na_filter = False) # empty cells are repesented as empty strings
    L = meals + allergens + dietary_restrictions
    matrix = pd.DataFrame(columns=L)
    
    for index, row in df.iterrows():
        item = []
        item.append(row["Food Item"])
        
    # ['Food Item','Breakfast', 'Lunch', 'Dinner']
        meal = row[' Meal']
        if meal == "Breakfast": item.append(1)
        else: item.append(0)

        # Lunch
        if meal == "Lunch": item.append(1)
        else: item.append(0)

        # Dinner
        if meal == "Dinner": item.append(1)
        else: item.append(0)

    # ['Fish', 'Soybeans', 'Wheat', 'Gluten', 'Milk', 'Tree Nuts', 'Eggs', 'Sesame', 'Crustacean Shellfish']
        allergen = row[' Allergens']
        if "Fish" in allergen: item.append(1)
        else: item.append(0)

        if  "Soybeans" in allergen: item.append(1)
        else: item.append(0)

        if  "Wheat" in allergen: item.append(1)
        else: item.append(0)

        if "Gluten" in allergen: item.append(1)
        else: item.append(0)

        if "Milk" in allergen: item.append(1)
        else: item.append(0)

        if "Tree Nuts" in allergen: item.append(1)
        else: item.append(0)

        if "Eggs" in allergen: item.append(1)
        else: item.append(0)

        if "Sesame" in allergen: item.append(1)
        else: item.append(0)

        if "Crustacean Shellfish" in allergen: item.append(1)
        else: item.append(0)

    # ['Gluten Friendly', 'Less Sodium', 'Smart Check', 'Vegan', 'Vegetarian', 'Contains Nuts']
        dietary_restriction = row[' Dietary Restrictions']
        if "Gluten Friendly" in dietary_restriction: item.append(1)
        else: item.append(0)

        if "Less Sodium" in dietary_restriction: item.append(1)
        else: item.append(0)

        if "Smart Check" in dietary_restriction: item.append(1)
        else: item.append(0)

        if "Vegan" in dietary_restriction: item.append(1)
        else: item.append(0)

        if "Vegetarian" in dietary_restriction: item.append(1)
        else: item.append(0)

        if "Contains Nuts" in dietary_restriction: item.append(1)
        else: item.append(0)

        item = pd.DataFrame([item], columns = meals + dietary_restrictions + allergens)
        matrix = pd.concat([matrix, item])

#     path = "./preprocessed-data/{}.csv".format(dining_hall)
#     matrix.to_csv(path, encoding='utf-8', index=False)

    return matrix


if __name__ == "__main__":
    print(preProcessData("whitney"))