from Tests import iris_boost, california_housing_boost, california_housing_van, iris_van


def main():
    california_housing_boost.run("exact", rounds=50)


if __name__ == "__main__":
    main()
