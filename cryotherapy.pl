get_result(Sex, Age, Time, Number_of_Warts, Type, Area):-
	 Time =< 8.125 -> 
		age_condition_one(Age); 
		age_condition_two(Age, Area, Type).
			
			
age_condition_one(Age) :-
	Age =< 52 -> 
			write("Success") ; 
			write("Failure").
			

age_condition_two(Age, Area, Type) :-
	Age =< 16.5 -> 
			write("Success") ; 
			area_condition_one(Age, Area, Type).

area_condition_one(Age, Area, Type) :-
	Area =< 27.5 -> 
			area_condition_two(Area, Type);
			age_condition_three(Age).
			
age_condition_three(Age) :-
	Age =< 20 -> 
			write("Success") ; 
			write("Failure").
			
area_condition_two(Area, Type) :-
	Area =< 15 ->
			write("Success") ;
			type_condition_one(Type).
			
type_condition_one(Type) :-
	Type =< 2.5 ->
			write("Success") ; 
			write("Failure").