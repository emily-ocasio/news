clear all

import delimited homicide_data2.csv
local strlist "cntyfips ori state agency agentype source solved month actiontype homicide situation vicsex vicrace vicethnic offsex offrace offethnic weapon relationship circumstance msa yearmonth vicrace_offrace"
foreach x of local strlist {
	encode `x', gen(`x'_e)
	drop `x'
	rename `x'_e `x'
}

*Recoding
replace vicrace = 0 if vicrace == 5
recode  vicrace (3 = 1) (1 = 3)
recode  offrace (3 = 1) (1 = 3)
replace offrace = 0 if offrace == 5

*Simplified victim race for summary statistics
recode vicrace (0 = 0) (1 = 1) (2/4 = 2), gen(vicrace_simple)

*Labeling
label define vicracelabel 0 "White" 1 "Black" 2 "  Asian" 3 "  AmerInd" 4 "  Unknown"
label define offracelabel 0 "White" 1 "Black" 2 "  Asian" 3 "  AmerInd" 4 "  Unknown"
label values vicrace vicracelabel
label values offrace offracelabel
label variable vicrace "Victim Race"
label define vicracesimplelabel 0 "White" 1 "Black" 2 "Other"
label values vicrace_simple vicracesimplelabel
label variable vicrace_simple "Victim Race"


***** Sex *****

*Recoding
recode vicsex (2=0) (3=2)
recode offsex (2=0) (3=2)

*Labeling
label define vicsexlabel 0 "Male" 1 "Female" 2 "Unknown"
label define offsexlabel 0 "Male" 1 "Female" 2 "Unknown"
label values vicsex vicsexlabel
label values offsex offsexlabel
label variable vicsex "Victim Sex"


***** Age *****

*Victim
gen vicage_c = 1 if vicage<18
replace vicage_c = 2 if vicage>=18 & vicage<30
replace vicage_c = 3 if vicage>=30 & vicage<50
replace vicage_c = 4 if vicage>=50 & vicage<70
replace vicage_c = 5 if vicage>=70 & vicage<999
replace vicage_c = 6 if vicage==999

*Offender
gen offage_c = 1 if offage<18
replace offage_c = 2 if offage>=18 & offage<30
replace offage_c = 3 if offage>=30 & offage<50
replace offage_c = 4 if offage>=50 & offage<70
replace offage_c = 5 if offage>=70 & offage<999
replace offage_c = 6 if offage==999

*Labeling
label define agelabel 1 "<18" 2 "18-30" 3 "30-49" 4 "50-69" 5 "70+" 6 "n/a"
label values vicage_c agelabel
label values offage_c agelabel
label variable vicage_c "Victim Age"


***** Weapon *****

*Recoding
recode weapon (6 7 10 15 16 = 1) (8 = 2) (2 = 3) (12 17 = 4) (1 3 4 5 9 11 13 14 = 5)

*Labeling
label  define weaponlabel 1 "Gun" 2 "Knife" 3 "Blunt object" 4 "Beating/strangulation" 5 "NEC"
label values weapon weaponlabel


***** Relationship *****

*Recoding
recode relationship (1 = 1) (2 11 = 2) (3 20 = 3) (4 5 13 26 = 4) (6 21 = 5) (7 = 6) (8 = 7) (9 15 = 8) (10 = 9) ///
				    (12 = 10) (14 = 11) (16 = 12) (17 = 13) (18 = 14) (19 = 15) (22 23 24 = 16) (25 = 17)

*Labeling
label  define relationshiplabel 1 "Aquiantance" 2 "Boyfriend/Girlfriend" 3 "Brother/Sister" 4 "Husband/Wife" 5 "Daughter/Son" 6 "Employee" 7 "Ex-Wife" 8 "Father/Mother" 9 "Friend" 10 "Homosexual relationship" 11 "In-law" 12 "Neighbor" 13 "Other - Known (not family)" 14 "Other - Known (family)" 15 "Unknown relationship" 16 "Step family" 17 "Stranger" 
label values relationship relationshiplabel


***** Circumstance *****

*Recoding
recode circumstance (1 = 1) (2 = 2) (3 22 = 3) (4 = 4) (5 6 = 5) (7 = 6) (8 9 18 20 21 25 = 7) (10 = 8) (12 = 9) (13 = 10) (14 16 = 11) (17 = 12) (19 = 13) (24 26 = 14) (27 = 15) (11 = 16) (15 = 17) (23 28 = 18)

*Labeling
label define circumstancelabel 1 "Manslaughter by negligence (NEC)" 2 "Suspected Felony Type" 3 "Argument" 4 "Arson" 5 "Brawl due to alcohol/narcotics" 6 "Burglary" 7 "Other" 8 "Undetermined" 9 "Felon killed by private citizen" 10 "Gambling" 11 "Gangs" 12 "Lovers triangle" 13 "Narcotic drug laws" 14 "Rape/other sex offense" 15 "Robbery" 16 "Felon killed by policy" 17 "Institutional killing" 18 "Accidental/negligent shooting"
label values circumstance circumstancelabel


***** Situation *****

gen multiplevic = (situation<4)
gen multipleoff = (situation==1|situation==4)


***** Dependant variables *****

*COMPOSITE humanizing score
gen humanizeall = humanized
replace humanizeall = 0 if humanized==.
label variable humanizeall "Composite Score"

*EXTENSIVE humanizing score
label variable assigncount     "Number of Articles"



*No interactions, no age
local controls_dem_nia 	i.vicrace i.vicsex  i.offrace i.offsex

*Demographic
local controls_dem 		i.vicage_c##i.vicrace##i.vicsex i.offage_c##i.offrace##i.offsex

*Demographic + Circumstantial
local controls_cir		i.weapon i.relationship i.circumstance i.incident i.agentype i.multiplevic i.multipleoff 

*Demographic + Circumstantial + Geography/Time
local controls_geo 		i.cntyfips  i.year#i.month

keep if vicrace<2 & vicage_c<5 & vicsex<2	


