

def make_breakfast(course_number, breakfast_in_bed):
    # Remove the incorrect re-initialization of breakfast_in_bed
    if course_number == 0:
        breakfast_in_bed[course_number] = ["eggs", "pancakes"]
    elif course_number == 1:
        breakfast_in_bed[course_number].extend(["coffee", "orange juice"])
    else:
        breakfast_in_bed[course_number].extend(["fruit crisp", "muffin"])

    return breakfast_in_bed


def main():
    # Create and populate breakfast_in_bed list in the proper for loop
    breakfast_in_bed = [[], [], []]
    for meal_course_counter in range(3):
        breakfast_in_bed = make_breakfast(meal_course_counter, breakfast_in_bed)

    print(breakfast_in_bed)


if __name__ == "__main__":
    main()
