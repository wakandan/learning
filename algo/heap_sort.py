import sys


arr = [int(i) for i in "5 25 15 8 7 28 1 4 10 9".split()]

def heapify(a, i):
    while i<len(a)/2:
        child1 = 2*i+1
        child2 = 2*i+2
        larger_child_index = child1
        if child2 < len(a) and a[child2] > a[larger_child_index]:
            larger_child_index = child2

        if a[i] < a[larger_child_index]:
            a[i], a[larger_child_index] = a[larger_child_index], a[i]
        i = larger_child_index

def build_heap(a):
    for i in range(len(a)-1, -1, -1):
        heapify(a, i)
        print(a)

def heap_sort(a):
    print('building heap')
    build_heap(a)
    print('sorting')
    for i in range(len(a)-1, -1, -1):
        print(a[0])
        a[0], a[i] = a[i], a[0]
        a = a[:-1] 
        heapify(a, 0)

heap_sort(arr)
