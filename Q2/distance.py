def distance(target, source, insertcost, deletecost, replacecost):
    n = len(target)+1
    m = len(source)+1
    
    # set up dist and initalize values
    dist = [ [0 for j in range(m)] for i in range(n) ]
    # set up for instruction table
    instr = [ [0 for j in range(m)] for i in range(n) ]
    for i in range(1,m):
        dist[i][0] = dist[i-1][0] + insertcost
    for j in range(1,m): 
        dist[i][0] = dist[i-1][0] + deletecost

    # align source and target strings
    for j in range(1, m):
        for i in range(1, n):
            inscost = insertcost + dist[i-1][j]
            delcost = deletecost + dist[i][j-1]
            if (source[j-1] == target[i-1]): 
                add = 0
            else:
                add = replacecost
            substcost = add + dist[i-1][j-1]                  
                
            dist[i][j] = min(inscost, delcost, substcost)
            
            if (dist[i][j] == substcost):
                instr[i][j] = 's'
            elif (dist[i][j] == delcost):
                instr[i][j] = 'd'
            else:
                instr[i][j] = 'i'
            
            
    alignment, count = visualize(target, source, dist, instr)
    
    # return min edit distance
    return dist[n-1][m-1], alignment, count

def visualize(target, source, dist, instr):
    top, mid, bot = 0, 1, 2    
    arr = [ [" " for j in range(max(len(target), len(source)) * 2)] for i in range(3) ]

    i = len(target)
    j = len(source)
    count = 0
    t = (max(len(target), len(source))*2) - 1
    while ((i!=0 and j!=-1) or (i!=-1 and j!=0)):
        inst = instr[i][j]
        if (j == 0):
            arr[top][t] = target[i-1]
            arr[bot][t] = "_"
            i = i - 1
        elif (i == 0):
            arr[top][t] = "_"
            arr[bot][t] = source[j-1]
            j = j - 1
        elif (inst == 'i'):
            # insertion
            arr[top][t] = target[i-1]
            arr[bot][t] = "_"
            i = i - 1
            
        elif (inst == 'd'):
            # deletion
            arr[top][t] = "_"
            arr[bot][t] = source[j-1]
            j = j - 1
        else:
            # substitution
            arr[top][t] = target[i-1]
            arr[bot][t] = source[j-1]
            if (target[i-1] == source[j-1]): 
                arr[mid][t] = '|'
            j = j - 1
            i = i - 1
        t = t -1
        count = count + 1
    return arr, count

def printAlignment(arr, count):
    for i in range(len(arr[0])):
        if (arr[0][i] != ' '):
            print(' ' + arr[0][i], end='')
    print()

    for i in range(len(arr[1]) - count, len(arr[1])):
        print(' ' + arr[1][i], end='')
    print()

    for i in range(len(arr[2])):
        if (arr[2][i] != ' '):
            print(' ' + arr[2][i], end='')

if __name__=="__main__":
    from sys import argv
    if len(argv) > 2:
        distance, alignment, count = distance(argv[1], argv[2], 1, 1, 2)
        print("levenshtein distance =", distance)  
        printAlignment(alignment, count)
