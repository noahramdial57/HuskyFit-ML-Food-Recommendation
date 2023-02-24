from csvStorage import storeData

# pass dining hall names in as lower case

try:
    storeData("northwest")
    print("Northwest Data has been stored.")
except:
    print("Northwest Data cannot be stored.")

try:
    storeData("gelfenbien") # towers
    print("Towers Data has been stored.")
except:
    print("Towers Data cannot be stored.")

try:
    storeData("kosher") # towers kosher kitchen
    print("Towers-Kosher Data has been stored.")
except:
    print("Towers-kosher Data cannot be stored.")

try:
    storeData("McMahon")
    print("McMahon Data has been stored.")
except:
    print("McMahon Data cannot be stored.")

try:
    storeData("north")
    print("North Data has been stored.")
except:
    print("North Data cannot be stored.")


try:
    storeData("putnam")
    print("Putnam Data has been stored.")
except:
    print("Putnam Data cannot be stored.")


try:
    storeData("south")
    print("South Data has been stored.")
except:
    print("South Data cannot be stored.")


try:
    storeData("whitney")
    print("Whitney Data has been stored.")
except:
    print("Whitney Data cannot be stored.")
