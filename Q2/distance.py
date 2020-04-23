

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
            
            if (dist[i][j] == inscost):
                instr[i][j] = 'i'
            elif (dist[i][j] == delcost):
                instr[i][j] = 'd'
            else:
                instr[i][j] = 's'
            
            
    visualize(target, source, dist, instr)
            
    # return min edit distance
    return dist[n-1][m-1]



def visualize(target, source, dist, instr):
    top, bot, mid = 0, 1, 2    
    arr = [ [" " for j in range(max(len(target), len(source)))] for i in range(3) ]
    
    print(len(arr[top]))
    # print(arr)
    # for i in range(len(target)):
    #     arr[top][i] = target[i]
        
    # for i in range(len(source))
    #     arr[bot][i] = source[i]
    
    i = len(target) -1
    j = len(source) -1 
    
    t = len(arr[top]) -1 
    while (t >= 0):
        inst = instr[i][j]
        
        if (inst == 'i'):
            # insertion - go left
            print(target[i])
            print(arr[top][t])
            arr[top][t] = target[i]
            arr[bot][t] = "_"
            i = i - 1
        elif (inst == 'd'):
            # deletion - go down
            arr[top][t] = target[i]
            arr[bot][t] = "_"
            j = j - 1
        else:
            # substitution - go diagonal
            arr[top][t] = target[i]
            arr[bot][t] = source[j]
            j = j - 1
            i = i - 1
        t = t -1
        
    print(arr[top])
    print(arr[mid])
    print(arr[bot])   

    
    

if __name__=="__main__":
    from sys import argv
    if len(argv) > 2:
        print("levenshtein distance =", distance(argv[1], argv[2], 1, 1, 2))  
        
            