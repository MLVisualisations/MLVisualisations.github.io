def bubblesort(array):
    for i in range(len(array)):
        for j in range(len(array) - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]

    return array

if __name__ == '__main__':
    array = [1, 4, 24, 21, 53, 102, 6, 42, 16, 99]

    array = bubblesort(array)
    print(array)
