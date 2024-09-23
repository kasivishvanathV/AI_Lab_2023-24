# Ex.No: 8  Logic Programming –  Medical Diagnosis Expert System
### DATE:        
23/09/2024
### REGISTER NUMBER : 
212222040073
### AIM: 
Write a Prolog program to build a medical Diagnosis Expert System.
###  Algorithm:
1. Start the program.
2. Write the rules for each diseases.
3. If patient have mumps then symptoms are fever and swollen glands.
4. If patient have cough, sneeze and running nose then disease is measles.
5. if patient have symptoms headache ,sneezing ,sore_throat, runny_nose and  chills then disease is common cold.
6. Define rules for all disease.
7. Call the predicates and Collect the symptoms of Patient and give the hypothesis of disease.
        

### Program:
```
hypothesis(Patient,german_measles) :-
	symptom(Patient,fever),
	symptom(Patient,headache),
	symptom(Patient,runny_nose),
	symptom(Patient,rash).
hypothesis(Patient,flu) :-
        symptom(Patient,fever),
       symptom(Patient,headache),
	symptom(Patient,body_ache),
	symptom(Patient,conjunctivitis),
	symptom(Patient,chills),
	symptom(Patient,sore_throat),
	symptom(Patient,runny_nose),
	symptom(Patient,cough).
hypothesis(Patient,common_cold) :-
	symptom(Patient,headache),
	symptom(Patient,sneezing),
	symptom(Patient,sore_throat).
hypothesis(Patient,chicken_pox) :-
	symptom(Patient,fever),
	symptom(Patient,chills),
	symptom(Patient,body_ache),
	symptom(Patient,rash).
hypothesis(Patient,measles) :-
	symptom(Patient,cough),
	symptom(Patient,sneezing),
	symptom(Patient,runny_nose).
symptom(raju,headache).
symptom(raju,sneezing).
symptom(raju,sore_throat).
```

### Output:
![Screenshot (147)](https://github.com/user-attachments/assets/aece8eb8-3143-4133-adc7-ed568f1847ca)
![Screenshot 2024-09-23 091900](https://github.com/user-attachments/assets/6c1cf6bf-f808-4240-aad2-c6f2d7a42704)
![Screenshot 2024-09-23 092138](https://github.com/user-attachments/assets/b249f022-54f6-4ff2-9aa2-c05ce2e7c026)




### Result:
Thus the simple medical diagnosis system was built sucessfully.
