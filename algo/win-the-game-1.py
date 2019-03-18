'''
# Sample code to perform I/O:

name = input()                  # Reading input from STDIN
print('Hi, %s.' % name)         # Writing output to STDOUT

# Warning: Printing unwanted or ill-formatted data to output will cause the test cases to fail
'''

# Write your code here
t = int(input())
for i in range(t):
    r, g = map(int, input().split())
    if r == 0 or g == 0:
        print("{0:.6f}".format(1))
        continue
    if g<2:
        print("{0:.6f}".format(r/(r+g)))
        continue
    p = [0 for j in range(g+1)]
    p[0] = 1
    p[1] = r/(r+g)
    for j in range(2, g+1):
        p[j] = r/(r+g) + g*(g-1)/((r+g)*(r+g-1))*p[j-2]
    print("{0:.6f}".format(p[g]))
