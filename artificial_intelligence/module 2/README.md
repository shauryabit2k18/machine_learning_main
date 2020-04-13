# Game Playing Agent: ISOLATION

![1](https://github.com/shauryabit2k18/udacity_artificial_intelligence/blob/master/module%202/1.gif)

The purpose of this project is to design and implement a game playing agent to play a game using adversarial search methods. The goal is to create a game playing agent that defeats our opponent at a game of isolation consistently. For this project I’ve designed three heuristics to create an effective edge for the game playing agent. 

## MinMax:

In the case of the game isolation, there are two players competing against each other. We are under the assumption that the AI player and human player are both playing two win. While back-propagating through the search tree, our algorithm must maximize the outcome (choose the optimal move) for when it’s the AI’s turn and minimize the outcome (choose the least optimal move for us) when it’s the human player’s turn.

![2](https://github.com/shauryabit2k18/udacity_artificial_intelligence/blob/master/module%202/2.png)

Observe the bottom-left portion of the search tree — the bottom nodes are maximizing nodes. Maximizing nodes select the highest value (most optimal outcome). As you may guess, minimizing nodes select the lowest value (least optimal outcome). Imagine that you are playing a game of chess, each move you make is a maximizing node (your intention is to win) — and every move your opponent makes is a minimizing node (move that messes up your game 😞).

## Iterative Deepening:

Thus far in the Nanodegree we’ve been exploring search algorithms that enable us to brute force winning outcomes based on a specific game state. Some games have a very large game-state space (same is true for isolation game) — which would make traditional depth-first search an inefficient solution to find a winning outcome. In some cases, we may want to search each possible outcome of a game move while returning an optimal next move in a reasonable time. That’s where iterative deepening comes into play. With iterative deepening, you define the minimum search depth; if time persists, another depth will be explored.

![3](https://github.com/shauryabit2k18/udacity_artificial_intelligence/blob/master/module%202/3.gif)

## Isolation Game Tree with Leaf Values

![5](https://github.com/shauryabit2k18/udacity_artificial_intelligence/blob/master/module%202/5.svg)

Take some time to study the game tree, then continue on.

Note the second subtree at Level 1 (where O picks the top, middle cell) - it leads to a lot of losses (-1 values).

![6](https://github.com/shauryabit2k18/udacity_artificial_intelligence/blob/master/module%202/6.png)

## Evaluation Functions:

Once the minimum search depth is reached, we must evaluate each node and provide a score. For example, an evaluation function could be the number of moves a player has available — The below figure shows an example search tree with the number of moves available in each game state.

![4](https://github.com/shauryabit2k18/udacity_artificial_intelligence/blob/master/module%202/4.png)

For this project I’ve designed and implemented three evaluation functions.
