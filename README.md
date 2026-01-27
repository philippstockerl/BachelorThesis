# BachelorThesis
This is the corresponding code for my bachelor thesis titled: "Robust and Adaptive Path Planning for Autonomous Vehicles in Spatio-Temporal Cost Fields".

My research objective was to evaluate different shortest-path algorithms on the same graph and cost structure.
For that I have implemented a directed graph generator using the Python library GSTools to simulate weather cells that resemble cost fields.
In my thesis I took a closer look at robust combinatorial optimization and incremental search methods.
My goal was to compare two very different approaches, or "paradigms", on how to handle edge cost uncertainty.
Robust optimization has many different general formulations on how to plan a minimum-cost shortest-path on a graph that hedge against worst-case costs before the actual uncertainty is revealed.
In comparison, incremental search methods, specifically D* Lite, discover uncertainty during execution time and adapt to changing environments.

The robust optimization approaches where implemented as Mixed Integer Linear Programs and solved with the Gurobi Python solver.

D* Lite was also implemented from pseudocode (see Likhachev & Koenig 2001, D* Lite) and extended to my problem and data structure.

In addition, I have used D* Lite's structure to implement an experimental hybrid of both approaches.
While I compare the two standalone paradigms, I also used the robust combinatorial shortest-path solutions to guide D* Lite.
This experimental hybrid shows that a combination of global and local planners can yield real benefits in terms of solution quality and runtime.
