def eat_and_open(dinner_and_gifts):
    """Populate the dinner and gifts lists."""
    
    for counter in range(2):
        if counter == 0:
            dinner_and_gifts[0].append("steak")
            dinner_and_gifts[0].append("twice-baked potato")
            dinner_and_gifts[0].append("chocolate pie")
        elif counter == 1:
            dinner_and_gifts[1].append("racquet")
            dinner_and_gifts[1].append("tennis balls")
            dinner_and_gifts[1].append("sweatband")
            
    # Return the populated list in the proper place
    return dinner_and_gifts

def main():
    """Main function to create and populate dinner and gifts lists."""
    dinner_and_gifts = [[], []]
    dinner_and_gifts = eat_and_open(dinner_and_gifts)
    
    print(dinner_and_gifts)


if __name__ == "__main__":
    main()
