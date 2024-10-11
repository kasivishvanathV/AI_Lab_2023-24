# Ex.No: 10  Logic Programming –  Simple queries from facts and rules
### DATE: 23/09/2024                                                                           
### REGISTER NUMBER: 212222040073
### AIM: 
To write a prolog program to find the answer of query. 
###  Algorithm:
 Step 1: Start the program <br> 
 Step 2: Convert the sentence into First order Logic  <br> 
 Step 3:  Convert the sentence into Horn clause form  <br> 
 Step 4: Add rules and predicates in a program   <br> 
 Step 5:  Pass the query to program. <br> 
 Step 6: Prolog interpreter shows the output and return answer. <br> 
 Step 8:  Stop the program.
### Program:
### Task 1:
Construct the FOL representation for the following sentences <br> 
1.	John likes all kinds of food.  <br> 
2.	Apples are food.  <br> 
3.	Chicken is a food.  <br> 
4.	Sue eats everything Bill eats. <br> 
5.	 Bill eats peanuts  <br> 
   Convert into clause form and Prove that John like Apple by using Prolog. <br> 
### Program:
```
:- discontiguous likes/2.
likes(john, X) :- food(X).
food(apples).
food(chicken).
eats(sue, X) :- eats(bill, X).
eats(bill, peanuts).
likes(john, apples).
```
### Output:
![image](https://github.com/Prathipa2003/AI_Lab_2023-24/assets/162797752/0152aaf3-32fc-45c0-a03e-d6c871574953)
### Task 2:
Consider the following facts and represent them in predicate form: <br>              
1.	Steve likes easy courses. <br> 
2.	Science courses are hard. <br> 
3. All the courses in Have fun department are easy <br> 
4. BK301 is Have fun department course.<br> 
Convert the facts in predicate form to clauses and then prove by resolution: “Steve likes BK301 course”<br> 
### Program:
```
likes(steve,X):-
     easycourse(X).
hard(sciencecourse).
easycourse(X):-
          course(X,dept(havefun)).
course(bk301,dept(havefun)).
```
### Output:
![image](https://github.com/Prathipa2003/AI_Lab_2023-24/assets/162797752/2bfdb66b-4563-4b15-81f9-72c15d2a3183)
### Task 3:
Consider the statement <br> 
“This is a crime for an American to sell weapons to hostile nations. The Nano , enemy of America has some missiles and its missiles were sold it by Colonal West who is an American” <br> 
Convert to Clause form and prove west is criminal by using Prolog.<br> 
### Program:
```
american(west).
enemy(nano, america).
owns(nano, missile).
criminal(X) :-
    american(X),
    weapon(Y),
    hostile(Z),
    sells(X, Y, Z).
hostile(nano).
sells(west, Y, nano) :-
    weapon(Y),
    owns(nano, Y).
weapon(missile).
```
### Output:
![image](https://github.com/Prathipa2003/AI_Lab_2023-24/assets/162797752/31ac8048-38ff-4cb1-bdda-3c6d0ea6250f)
### Result:
Thus the prolog programs were executed successfully and the answer of query was found.
