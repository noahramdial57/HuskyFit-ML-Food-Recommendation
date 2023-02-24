from dining import getMeals
from csv import writer
import sys

# Food Item, Dining Hall, Meal, Allergens, Dietary Restrictions, Date, Calories, Serving Size, Total Fat, Total Carbohydrates, Saturated Fat, Dietary Fiber , Trans Fat, Total Sugars, Cholesterol, Added Sugars, Sodium, Protein, Calcium, Iron, Vitamin D, Potassium

def storeData(dining_hall): 

    breakfast = getMeals("Breakfast", dining_hall.capitalize())
    lunch = getMeals("Lunch", dining_hall.capitalize())
    dinner = getMeals("Dinner", dining_hall.capitalize())

    with open('{}.csv'.format(dining_hall), 'a', newline='') as f_object:

        writer_object = writer(f_object)

        for item in breakfast:
            writer_object.writerow(item)

        for item in lunch:
            writer_object.writerow(item)

        for item in dinner:
            writer_object.writerow(item)

