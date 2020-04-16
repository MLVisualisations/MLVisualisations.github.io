% Unsorted array.
unsorted_array = [1, 4, 24, 21, 53, 102, 6, 42, 16, 99]

% Execute Bubble Sort function to function to sort array.
sorted_array = bubble_sort(unsorted_array)


% Bubble Sort as sub-function.
function array = bubble_sort( array )
%{
Sorts the entries of the vector 'array' into ascending order using the
Bubble Sort algorithm.

Inputs:
-------
array : vector
    List of numbers (un-ordered).

Outputs:
--------
array : vector
    List of numbers ordered by size, lowest to highest.
%}

% Pass through the array.
for i = 1:length(array)
    % Pass up to the last un-sorted element.
    for j = 1:length(array)-i
        % If elements are in the wrong order, swap them.
        if(array(j)>array(j+1))
            temp = array(j+1);
            array(j+1) = array(j);
            array(j) = temp;
        end
    end
end

end


