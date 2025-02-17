
import numpy as np

def neville_method():
    # Data
    xi = np.array([3.6, 3.8, 3.9], dtype=float)
    fxi = np.array([1.675, 1.436, 1.318], dtype=float)
    xp = 3.7

    # Set up the table with 0 first
    n = len(xi)
    table1 = np.zeros((n, n))
    
    # Set up the first column with f(x)
    for i in range(n):
        table1[i, 0] = fxi[i]

    # Calculate the interpolating values
    for i in range(1, n):
        for j in range(1, i + 1):
            num = (xp - xi[i - j]) * table1[i, j - 1] - (xp - xi[i]) * table1[i - 1, j - 1]
            den = xi[i] - xi[i - j]
            table1[i, j] = num / den

    # Print the 2nd degree interpolating value
    print(table1[2, 2])
    
    print()


def newton_foward_method():
    # Data
    xi = np.array([7.2, 7.4, 7.5, 7.6], dtype=float)
    fxi = np.array([23.5492, 25.3913, 26.8224, 27.4589], dtype=float)

    # Set up the table with 0 first
    n = len(xi)
    table2 = np.zeros((n, n))
    
    # Set up the first column with f
    for i in range(n):
        table2[i, 0] = fxi[i]

    # Calculate the divided differences
    for i in range(1, n):
        for j in range(1, i + 1):
            table2[i, j] = (table2[i, j - 1] - table2[i - 1, j - 1]) / (xi[i] - xi[i - j])

    # Print the coefs
    for i in range(1, n):
        print(table2[i, i])
    
    print()

    # Data
    xp = 7.3
    
    # Calculate the approximations
    p1 = fxi[0] + table2[1, 1]*(xp - xi[0])
    p2 = p1 + table2[2, 2]*(xp - xi[0])*(xp - xi[1])
    p3 = p2 + table2[3, 3]*(xp - xi[0])*(xp - xi[1])*(xp - xi[2])
    
    print(p3)
    
    print()


def hermite_matrix():
    # Data
    xi = [3.6, 3.6, 3.8, 3.8, 3.9, 3.9]
    fxi = [1.675, 1.675, 1.436, 1.436, 1.318, 1.318]
    
    n = len(xi)
    table3 = np.zeros((n, n - 1)) 

    # Set up the table with first column is x and second column is f
    for i in range(n):
        table3[i][0] = xi[i]
        table3[i][1] = fxi[i]

    # Calculate the divided differences
    # first order
    fd = [-1.195, -1.195, -1.188, -1.188, -1.182, -1.182]
    for i in range(1, n):
        # Ex: if x1 = x2, set x2 = fd2
        if xi[i] == xi[i - 1]: 
            table3[i][2] = fd[i] 
        else:
            table3[i][2] = (table3[i][1] - table3[i - 1][1]) / (xi[i] - xi[i - 1])

    # 2+ order
    for j in range(3, n - 1):
        for i in range(j - 1, n):
            table3[i][j] = (table3[i][j - 1] - table3[i - 1][j - 1]) / (xi[i] - xi[i - j + 1])

    # Print the table
    np.set_printoptions(linewidth=200)
    print(table3)
    
    print()


def cubic_spline_interpolation():
    # Data
    xi = np.array([2, 5, 8, 10], dtype=float)
    fxi = np.array([3, 5, 7, 9], dtype=float)

    # Set up h array with 0 first and fill with h values 
    n = len(xi)
    h = np.zeros(n - 1, dtype=float)
    for i in range(n - 1):
        h[i] = xi[i + 1] - xi[i]

    # Construct matrix A
    A = np.zeros((n, n), dtype=float)
    
    # Start coef and end coef of A is 1
    A[0, 0] = A[n - 1, n - 1] = 1.0 

    # Calculate coefs of A
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]

    # Set up b with 0 and fill values
    b = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        b[i] = (3.0 / h[i]) * (fxi[i + 1] - fxi[i]) - (3.0 / h[i - 1]) * (fxi[i] - fxi[i - 1])

    # Set up and calculate solution for Ax = b using linear algebra solver
    x = np.linalg.solve(A, b)

    print(A) # Print A

    print(b) # Print b

    # Print vector c
    print("[0.", end="")
    for i in x[1:-1]:
        print(f" {i:.8f}", end="")
    print(" 0.]")

def main():
    neville_method() # Q1
    newton_foward_method() #Q2 + Q3
    hermite_matrix() # Q4
    cubic_spline_interpolation() # Q5

if __name__ == "__main__":
    main()
