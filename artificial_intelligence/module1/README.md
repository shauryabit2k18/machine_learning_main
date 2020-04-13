# AI Nanodegree: Project 1

## Sudoku Solver
![1](https://github.com/shauryabit2k18/udacity_artificial_intelligence/blob/master/module1/Capture.PNG)

In Project 1 of Artificial Intelligence Nanodegree we leverage techniques such as constraint propagation and depth first search search to solve sudoku puzzles. I’m assuming that everyone reading this post understands the rules of sudoku — If not, don’t worry, I was not completely familiar with the game before the project.

## Setting UP and Encoding the board
The first task of the project is to assign variables and write functions that can encode the board. Long story short, the goal is to write a function that can take an 81 character input and store the column/row and box value in a Python dictionary. For example,
![2](https://github.com/shauryabit2k18/udacity_artificial_intelligence/blob/master/module1/Capture2.PNG)

Fortunately, the project provided a function that displays the sudoku board once we have it properly encoded.

![3](https://github.com/shauryabit2k18/udacity_artificial_intelligence/blob/master/module1/Capture3.PNG)

The next step is to assign a value to the boxes that have a nonnumerical value. More specifically, replace all boxes that contain ‘.’ with ‘123456789’.

## Constraint Propagation
Constraint Propagation is a constraint satisfaction technique that reduces search space by using local constraints. As mentioned in the first paragraph, we are going to implement depth first search so reducing the search size is very important for fast and efficient puzzle solving. For this sudoku project, each box has constraints (I kind of think of them as rules) that can lead us to a solved sudoku puzzle.

## Constraint: Elimination
A rule in Sudoku is that a box cannot have the same value as one of it’s peers (other boxes in it’s 3x3 square, row, and column). We can begin to use this rule as a constraint and eliminate possibilities with the boxes that have ‘123456789’ as their value.

![4](https://github.com/shauryabit2k18/udacity_artificial_intelligence/blob/master/module1/Capture4.PNG)

Writing a function for this constraint instantly reduces the values on the board (See Below).

![5](https://github.com/shauryabit2k18/udacity_artificial_intelligence/blob/master/module1/Capture5.PNG)

Obviously, this technique does not reduce the values of all the boxes to a single possibility so we must employ other strategies.

## Constraint: Only Choice
To reduce the possible combinations of each box further we employ only choice strategy as a constraint. This strategy works by searching for boxes that have a possible combination that are not seen by other peers.

![6](https://github.com/shauryabit2k18/udacity_artificial_intelligence/blob/master/module1/Capture6.PNG)

The 3rd box in column three contains [1,4,7]. When observing other boxes in the 3x3 square you can notice that none of them have 1 as a possibility. Therefore, the 3rd box in column 3 has to be 1.
We can now use elimination and iteratively use only choice to solve the sudoku puzzle. To quickly and efficiently solve the puzzle we implement depth first search.

## Search
For this project we leveraged Depth First Search to solve a sudoku puzzle. This search algorithm leverages the reduced size of the puzzle returned from eliminate and only choice, then brute forces the last possibilities.

![7](https://github.com/shauryabit2k18/udacity_artificial_intelligence/blob/master/module1/Capture7.PNG)

Udacity provides excellent images to visualize depth first search. With this search technique we are essentially iterating through all the reduced possibilities to solve the sudoku puzzle (Starting with boxes with the least amount of possibilities first).

## Constraint: Naked Twins
One of the requirements of project 1 was to implement the naked twins sudoku strategy. More specifically, when a box has two possible combinations and shares the same possible combinations with another peer (hence naked twins), either of those combination values can be removed from other peers. In the puzzle below two peers have and share the same two possible combinations [2,3].

![8](https://github.com/shauryabit2k18/udacity_artificial_intelligence/blob/master/module1/Capture8.PNG)

Naked twins strategy enables us to eliminate 2 and 3 from the other peers in this column.

![9](https://github.com/shauryabit2k18/udacity_artificial_intelligence/blob/master/module1/Capture9.PNG)

