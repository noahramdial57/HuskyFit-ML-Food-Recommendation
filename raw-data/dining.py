from bs4 import BeautifulSoup as soup
from datetime import date
import requests
import unicodedata

def getMeals(meal, dining_hall):

    link_list= ["http://nutritionanalysis.dds.uconn.edu/longmenu.aspx?sName=UCONN+Dining+Services&locationNum=03&locationName=Buckley+Dining+Hall&naFlag=1&WeeksMenus=This+Week%27s+Menus&dtdate=",                    # 10%2f25%2f2022&mealName=Breakfast"
                "http://nutritionanalysis.dds.uconn.edu/longmenu.aspx?sName=UCONN+Dining+Services&locationNum=42&locationName=Gelfenbien+Commons%2c+Halal+%26+Kosher&naFlag=1&WeeksMenus=This+Week%27s+Menus&dtdate=",  # 10%2f27%2f2022&mealName=Breakfast"
                "http://nutritionanalysis.dds.uconn.edu/longmenu.aspx?sName=UCONN+Dining+Services&locationNum=05&locationName=McMahon+Dining+Hall&naFlag=1&WeeksMenus=This+Week%27s+Menus&dtdate=",                    # 10%2f27%2f2022&mealName=Breakfast"
                "http://nutritionanalysis.dds.uconn.edu/longmenu.aspx?sName=UCONN+Dining+Services&locationNum=07&locationName=North+Campus+Dining+Hall&naFlag=1&WeeksMenus=This+Week%27s+Menus&dtdate=",                 # 10%2f27%2f2022&mealName=Breakfast"
                "http://nutritionanalysis.dds.uconn.edu/longmenu.aspx?sName=UCONN+Dining+Services&locationNum=15&locationName=Northwest+Marketplace&naFlag=1&WeeksMenus=This+Week%27s+Menus&dtdate=",                # 10%2f27%2f2022&mealName=Breakfast
                "http://nutritionanalysis.dds.uconn.edu/longmenu.aspx?sName=UCONN+Dining+Services&locationNum=06&locationName=Putnam+Dining+Hall&naFlag=1&WeeksMenus=This+Week%27s+Menus&dtdate=",                      # 10%2f27%2f2022&mealName=Breakfast"
                "http://nutritionanalysis.dds.uconn.edu/longmenu.aspx?sName=UCONN+Dining+Services&locationNum=16&locationName=South+Campus+Marketplace&naFlag=1&WeeksMenus=This+Week%27s+Menus&dtdate=",                 # 10%2f27%2f2022&mealName=Breakfast"
                "http://nutritionanalysis.dds.uconn.edu/longmenu.aspx?sName=UCONN+Dining+Services&locationNum=01&locationName=Whitney+Dining+Hall&naFlag=1&WeeksMenus=This+Week%27s+Menus&dtdate=",
                "http://nutritionanalysis.dds.uconn.edu/longmenu.aspx?sName=UCONN+Dining+Services&locationNum=42&locationName=Gelfenbien+Commons%2c+Halal+%26+Kosher&naFlag=1&WeeksMenus=This+Week%27s+Menus&dtdate="]                    # 10%2f27%2f2022&mealName=Breakfast"

    # string issues with mcmahon
    if dining_hall == "Mcmahon":
        dining_hall = "McMahon"

    # What dining hall is it?
    url = ""
    for link in link_list:
        if dining_hall in link:
            url = link
            break

    # Refactor link | Get todays link
    today = date.today()
    dd = today.strftime("%d")
    mm = today.strftime("%m")
    yyyy = today.strftime("%Y")

    # Use this to hardcode dates
    # today = "2/22/2023"
    # mm = "02"
    # dd = "22"
    # yyyy = '2023'

    edited_url = url + mm + '%2f' + dd + '%2f' + yyyy + '&mealName=' + meal
    buckley = requests.get(edited_url)

    # HTML Parsing
    html = buckley.content
    page_soup = soup(html, "html.parser")

    # Grabs each menu_item
    containers = page_soup.findAll("div", {"class": "longmenucoldispname"})

    L = []
    for menu_item in containers:
        href = menu_item.a['href']
        new_url = "http://nutritionanalysis.dds.uconn.edu/" + href
        label = requests.get(new_url)

        # HTML Parsing
        html2 = label.content
        page_soup = soup(html2, "html.parser")
        nut_facts = set(page_soup.findAll("span", {"class": "nutfactstopnutrient"}))

        calories_val = page_soup.find("td", {"class": "nutfactscaloriesval"})
        serving_size = page_soup.findAll("div", {"class": "nutfactsservsize"})
        allergens = page_soup.find("span", {"class": "labelallergensvalue"})

        myList = [None] * 22
        myList[0] = menu_item.text # myList.append(("Food item", menu_item.text))
        myList[1] = dining_hall # myList.append(("Dining Hall", dining_hall))
        myList[2] = meal # myList.append(("Meal", meal))

        # For menu items with no corresponding Nutrition Labels | Don't include them
        try: 
            myList[3] = allergens.text #cmyList.append(("Allergens", allergens.text))
        except:
            break

        # Get all dietary restrictions
        restrictions = []
        for foo in page_soup.find_all('img', alt=True):
            text = foo['alt']
            if "Gluten Friendly" in text:
                restrictions.append("Gluten Friendly")
                continue
            if "Less Sodium" in text:
                restrictions.append("Less Sodium")
                continue
            if "Smart Check" in text:
                restrictions.append("Smart Check")
                continue

            restrictions.append(text)

        myList[4] = restrictions         # myList.append(("Dietary Restrictions", restrictions))
        myList[5] = str(today)           # myList.append(("Date", str(today)))
        myList[6] = calories_val.text    # myList.append(("Calories", calories_val.text))
        myList[7] = serving_size[1].text # myList.append(("Serving Size", serving_size[1].text))

        # print(nut_facts)

        for facts in nut_facts:

            clean_text = unicodedata.normalize("NFKD", facts.text).strip() # remove unicode and leading spaces
            if clean_text.split() == []: 
                pass
            else:
                parse = clean_text.split()[-1] # get grams of nutritional fact

            # parse out weird strings
            if "%" in clean_text or "g" not in clean_text:
                continue

            # print(type(clean_text))
            # print()

            if "Total Fat" in clean_text:
                myList[8] = parse # myList.append(("Total Fat", parse))
                continue
            
            if "Total Carbohydrate." in clean_text:
                # print("here")
                myList[9] = parse # myList.append(("Total Carbohydrates", parse))
                continue

            if "Saturated Fat" in clean_text:
                myList[10] = parse # myList.append(("Saturated Fat", parse))
                continue

            if "Dietary Fiber" in clean_text:
                myList[11] = parse # myList.append(("Dietary Fiber", parse))
                continue

            if "Trans Fat" in clean_text:
                myList[12] = parse # myList.append(("Trans Fat", parse))
                continue

            if "Total Sugars" in clean_text:
                myList[13] = parse # myList.append(("Total sugars", parse))
                continue

            if "Cholesterol" in clean_text:
                myList[14] = parse # myList.append(("Cholesterol", parse))
                continue

            if "Added Sugars" in clean_text:
                parse = clean_text.split()[1] # get grams of nutritional fact
                myList[15] = parse                 # myList.append(("Added Sugars", parse))
                continue

            if "Sodium" in clean_text:
                myList[16] = parse # myList.append(("Sodium", parse))
                continue

            if "Protein" in clean_text:
                myList[17] = parse # myList.append(("Protein", parse))
                continue

            if "Calcium" in clean_text:
                myList[18] = parse # myList.append(("Calcium", parse))
                continue

            if "Iron" in clean_text:
                myList[19] = parse # myList.append(("Iron", parse))
                continue

            if "Vitamin D" in clean_text:
                myList[20] = parse # myList.append(("Vitamin D", parse))
                continue

            if "Potassium" in clean_text:
                myList[21] = parse # myList.append(("Potassium", parse))
                continue

            
        L.append(myList)

    return L