def bubblesort(array):
    """
    Inputs      : array (list)

    Outputs     : array (list) - sorted lowest to highest

    Description : sorts the array 'array' from lowest to highest
                  using bubblesort algorithm
    """

    # have to check at most length of array
    for i in range(len(array)):
        # loop over adjacent elements, up to already sorted elements
        for j in range(len(array) - i - 1):
            # checks for bigger number, if passed we swap the elements
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
    # once loop has exited the array has been sorted
    return array

if __name__ == '__main__':
    # use example from blog for array list
    unsorted_array = [1, 4, 24, 21, 53, 102, 6, 42, 16, 99]
    # execute bubblesort sorting function to sort array
    sorted_array = bubblesort(unsorted_array)
    # print to console for comparison 
    print('Unsorted array: ', unsorted_array)
    print('Sorted   array: ', sorted_array)
