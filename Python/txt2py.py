def txt2py(filename):
    with open(filename, 'r') as MyFile:
        A = []
        cont = 0
        for line in MyFile:
            A.append(line.split(';'))
            A[cont].pop()
            cont += 1
    return [map(float, A[i]) for i in range(len(A))]