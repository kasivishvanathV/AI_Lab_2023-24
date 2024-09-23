# Ex.No: 6   Logic Programming – Factorial of number   
### DATE:    
23/09/2024
### REGISTER NUMBER : 
212222040073
### AIM: 
To  write  a logic program  to solve Towers of Hanoi problem  using SWI-PROLOG. 
### Algorithm:
1. Start the program
2.  Write a rules for finding solution of Towers of Hanoi in SWI-PROLOG.
3.  a )	If only one disk  => Move disk from X to Y.
4.  b)	If Number of disk greater than 0 then
5.        i)	Move  N-1 disks from X to Z.
6.        ii)	Move  Nth disk from X to Y
7.        iii)	Move  N-1 disks from Y to X.
8. Run the program  to find answer of  query.

### Program:
```
move(1,X,Y,_) :-  
    write('Move top disk from '), 
    write(X), 
    write(' to '), 
    write(Y), 
    nl. 
move(N,X,Y,Z) :- 
    N>1, 
    M is N-1, 
    move(M,X,Z,Y), 
    move(1,X,Y,_), 
    move(M,Z,Y,X).
```
### Output:
![Screenshot (146)](https://github.com/user-attachments/assets/4c30dbcc-b2b0-44d9-a787-e70992a6ed63)
![Screenshot 2024-09-23 091729](https://github.com/user-attachments/assets/3359e792-e63c-4d42-9814-41f449d8f194)



### Result:
Thus the solution of Towers of Hanoi problem was found by logic programming.
