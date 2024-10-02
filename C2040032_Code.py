import sys
import math
import random
import time
import copy
#import csv
#from datetime import datetime

# dict of arrays used to for storing the tiings of each stages for efficiency testing
timings = {
    "fileRead" : [],
	"kemenyScoreTime" : [],
	"updateScoresTime" : []
}

def readData(fname):
    """
    Read data from the specified file and extract information about drivers and preferences.

    Parameters:
    - fname (str): The name of the file to read.

    Returns:
    - tuple: A tuple containing a dictionary of drivers, metadata, and a list of weights.
    """
    start = time.time()
    try:
        with open(fname, 'r') as f:
            # First line of the input file contains the number of drivers
            numDrivers = int(f.readline())

            # Generate a dictionary of driver numbers and names as key-value pairs with whitespace removed
            drivers = {}
            for _ in range(numDrivers):
                driverInfo = f.readline()
                dnumber, dname = driverInfo.split(",")
                drivers[int(dnumber)] = dname.strip()

            # Skip lines not important for coursework
            extraLine = f.readline().strip()
            numR, numPref, numPairWise = extraLine.split(",")
            metaData = {
                "Num Races": numR,
                "Num Preferences": numPref,
                "Num Pairwise Matchups": numPairWise
            }

            weights = []
            
            while True:
                weightLine = f.readline().strip()
                # Check for the empty final line
                if not weightLine:
                    print("Final data line found")
                    break
                else:
                    weightLine = weightLine.split(",")
                    # Create a list with starting vertex and a tuple of the weight and end vertex
                    edge = [int(weightLine[0]), (int(weightLine[1]), int(weightLine[2]))]
                    weights.append(edge)

        timings["fileRead"].append((time.time() - start) * 1000)

        return drivers, metaData, weights

    except FileNotFoundError:
        print(f"No file found with the name: {fname}. Please try again.")
    except Exception as e:
        print(f"An error occurred: {e}")

    pass


def genMat(weights, numDrivers):
    """
    Generate a matrix based on the given weights and the number of drivers.

    Parameters:
    - weights (list): A list of weight entries, each containing a weight value and an edge.
    - numDrivers (int): The number of drivers.

    Returns:
    - list: A 2D matrix representing the weights between drivers.
    """
    # Initialize an empty matrix with zeros
    matrix = [[0] * numDrivers for _ in range(numDrivers)]

    # Insert the values of each weight into the corresponding location in the matrix
    for weightEntry in weights:
        weight, edge = weightEntry[0], weightEntry[1]
        matrix[edge[0] - 1][edge[1] - 1] = weight

    return matrix

def individualKscore(mat, r, x):
    """
    Calculate the individual Kemeny score for a single driver.

    Parameters:
    - mat (list): The matrix of weights representing pairwise matchups.
    - r (list): The ranking of drivers.
    - x (int): The index of the driver for which the Kemeny score is calculated.

    Returns:
    - int: The Kemeny score for the specified driver.
    """

    start = time.time()
    # Generate scores for driver x by summing weights 
    # where driver x is ranked higher than other drivers y
    driverScore = sum([mat[r[y] - 1][r[x] - 1] for y in range(x, len(r))])
    timings["kemenyScoreTime"].append((time.time() - start) * 1000000)
    
    return driverScore

def kScores(mat, r):
    """
    Calculates the Kemeny scores for all drivers in a given ranking.

    Parameters:
    - mat (list): The matrix of weights representing pairwise matchups.
    - r (list): The ranking of drivers.

    Returns:
    - scores (list): A list of Kemeny scores for each driver in the ranking.
    """

    scores = []
    # Loops through all rankings and calls for the kemeny score of each individual driver
    for x in range(len(r)):
        driverScore = individualKscore(mat, r, x)
        # Append the driver Kemeny score to the list of all drivers
        scores.append(driverScore)

    return scores


def updateScores(indexA, indexB, sol, cost, mat):
    """
    Updates the scores in by calculating the updated scores between the changed
    neighbourhood indexs

    Paramteres:
    - indexA (int) : starting swap index
    - indexB (int) : ending swap index
    - sol (list) : current ranking
    - cost (list) : current Kemeny scores
    - mat (2D list) : matrix of driver weigths

    Returns:
    - updatedScores (list): updated Kememy scores
    """
    # Handle wrap around
    if indexA < 0:
        indexA = len(sol) - 1
    if indexB < 0:
        indexB = len(sol) - 1
  
    # Get driver IDs
    driverA = sol[indexA] 
    driverB = sol[indexB]
    
    # Swap drivers in ranking
    sol[indexA] = driverB  
    sol[indexB] = driverA

    # define start and end markers
    start = min(indexA, indexB)
    end = max(indexA, indexB) + 1

    newCosts = []
    # Recalculate scores in range     
    for i in range(start, end):
        newCosts.append(individualKscore(mat, sol, i))

    # Slice original scores to insert updated
    updatedScores = cost[:start] + newCosts + cost[end:]
    
    return updatedScores

def sa(TI, TL, cooling, cost, numNonImprove, initialSol, mat, nSize):
    """
    !!!!! THIS SECTION IS SIMULATED ANNEALING IMPLEMENTATION !!!!!
    Implementation of Simulated Annealing algorithm

    Paramaters:
    - TI (int): initial temperature
    - TL (int): temperature length
    - cooling (int): cooling ratio
    - cost (list): initial kemeny scores
    - numNonImprove (int): stopping criteria
    - initialSol (list): starting rankings of drivers
    - mat (2D list): weights martrix for drivers
    - nSize (int): maximum distance of random neighbour swap

    Returns: 
    - bestSol (int): optimum ranking
    - minCost (int): lowest kemeny score
    - upCount (int): number of uphill moves during the procedure
    """
    curCost = cost[:]
    minCost = cost[:]
    curSol = initialSol[:]
    bestSol = initialSol[:]
    T = TI
    upCount = 0
    moveSinceBest = 0
    while moveSinceBest < numNonImprove:
        for _ in range(TL):
            newSol = curSol[:]
            totalCost = sum(curCost)

            # Neighborhood change implementation
            indexA = random.randint(0, len(newSol) - 1 - nSize)
            options = [x for x in range(indexA, len(newSol) - 1)]
            indexB = random.sample(options, 1)[0] 

            start = time.time()
            # update the rankings and return new cost
            newCost = updateScores(indexA, indexB, newSol, curCost, mat)
            timings["updateScoresTime"].append((time.time() - start) * 1000000)

            newTotalCost = sum(newCost)

            if newTotalCost > totalCost:
                # probabilistic element of algorirthm
                delta = newTotalCost - sum(curCost)
                probability = math.exp(-delta / T)

                if random.uniform(0, 1) < probability:
                    curCost = newCost
                    curSol = newSol
                    upCount += 1
                    moveSinceBest += 1
            else:
                curCost = newCost
                curSol = newSol
                moveSinceBest += 1

                # if current solution is best so far, it becomes the best
                if sum(minCost) > sum(curCost):
                    bestSol = curSol
                    minCost = curCost
                    moveSinceBest = 0

        #multiply temperature by cooling value
        T *= cooling

    return bestSol, minCost, upCount


def testingVars(mat, ranking, kemenyScores):
    # lists of variable values to test
    TIl = [1, 2, 5, 10, 20, 40, 50, 100]
    TLl = [1, 2, 5, 10, 30, 50, 100, 150]
    coolingl = [0.9999, 0.9995, 0.99, 0.95, 0.90, 0.80]
    numNonImprovel = [100, 1000, 2000, 5000, 10000, 20000, 50000] 

    results = []
    i = 0

    for TI in TIl:
        for TL in TLl:
            for cooling in coolingl:
                for numNonImprove in numNonImprovel:
                        # Run SA with params
                        start = time.time()
                        bestRank, bestCost, upMoves = sa(TI, TL, cooling, kemenyScores, numNonImprove, ranking, mat, nSize=3)  

                        # Record parameters and cost 
                        result = [TI, TL, cooling, numNonImprove, sum(bestCost), upMoves, round((time.time() - start) * 1000, 2)]
                        results.append(result)
                
                # Write results  
                # # this is done incrementally to multiple files as the memory usage of storing all the variables and data
                # # crashed my machie a few times         
                with open('Data/sa_results' + "-" + str(i) + ".csv", 'w', newline="") as f:
                    writer = csv.writer(f) 
                    writer.writerow(['Init Temp', 'Temp length', 'Cooling', 'num non improve', 'Cost', "up moves", "time"])  
                    #flat_results = [item for sublist in results for item in sublist] 
                    writer.writerows(results)
                    current_time = datetime.now()
                    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"{formatted_time} --- test written")
                    i += 1
                results = []


def main():
    fname = sys.argv[1]
    #read the data from filename given
    drivers, metaData, weigths = readData(fname)

    # generate the matrix, intial ranmkings and kemeny scores
    mat = genMat(weigths, len(drivers))
    ranking = list(range(1, len(drivers) + 1))
    kemenyScores = kScores(mat, ranking)

    # deepcopy of rankings used to display the change in rankings at output
    originalRank = copy.deepcopy(ranking)
    
    #chosen optimal solutions
    TI = 1
    TL = 50    
    cooling = 0.9999
    numNonImprove = 2000

    #n size defines the maximum distance between drivers in the ranking to swap
    nSize = 3
    """
    !!! Used to test the most suitable variable configuration
    testingVars(mat, ranking, kemenyScores)
    """
    
    startTime = time.time()
    bestRanking, bestCost, upMoves = sa(TI, TL, cooling, kemenyScores, numNonImprove, ranking, mat, nSize)
    timeTaken = time.time() - startTime

    # formatted output of optimal solution statistics
    print("New Rank \tOriginal Rank\tDriver Name")
    for i, new_rank in enumerate(bestRanking):

        original_rank = originalRank[i]
        driver_name = drivers[new_rank]
        print(f"{original_rank} \t\t{new_rank}\t\t{driver_name}")

    print("\nOverview")
    print("Best Kemeny Score:\t", sum(bestCost))
    print("Up moves:\t\t", upMoves)
    print(f"Time taken: \t\t {round(timeTaken, 2) * 1000} milliseconds")

    print("\nDetailed time breakdown (average time for each procedure)")
    print(f"Read File       --- \t{round(sum(timings['fileRead']) / len(timings['fileRead']), 2)} (ms)")
    print(f"K Score         --- \t{round(sum(timings['kemenyScoreTime']) / len(timings['kemenyScoreTime']), 2)} (ns)")
    print(f"Update Score    --- \t{round(sum(timings['updateScoresTime']) / len(timings['updateScoresTime']), 2)} (ns)")
    

if __name__ == '__main__':  
    main()
