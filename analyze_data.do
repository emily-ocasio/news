*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ analyze_data.do   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

*Last updated: 12/28/2022





*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Preliminaries ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

*Stata stuff
set more off, permanently													
clear all

*Set directory
cd "/Users/`c(username)'/Dropbox/Research/Race and Media Coverage of Homicide/Data Analysis/"

*Get unemployment data
import excel MAUR, cellrange(A11:B119) firstrow
rename MAUR unrate
gen month=month(observation_date)
gen year=year(observation_date)
save MAUR, replace
clear all

*Get FBI/BG data
import delimited homicide_data2.csv
local strlist "cntyfips ori state agency agentype source solved month actiontype homicide situation vicsex vicrace vicethnic offsex offrace offethnic weapon relationship circumstance msa yearmonth vicrace_offrace"
foreach x of local strlist {
	encode `x', gen(`x'_e)
	drop `x'
	rename `x'_e `x'
}
joinby year month using MAUR

*Graph Settings
graph drop _all
graph set eps fontface "Times New Roman"
grstyle clear
set scheme s2color
grstyle init
grstyle set plain, box
grstyle color background white
grstyle set color Set1
grstyle yesno draw_major_hgrid yes
grstyle yesno draw_major_ygrid yes
grstyle color major_grid gs8
grstyle linepattern major_grid dot
grstyle set legend 4, box inside
grstyle color ci_area gs12%50


	
	

*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~ Variables for analysis ~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

***** Race *****

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





*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Summary statistics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*


***** Figure: Article count histogram *****
scalar maxval = 20
gen 	assigncount_max = assigncount if assigncount<=maxval
replace assigncount_max = maxval if assigncount>maxval
label variable assigncount_max "Number of Articles"
histogram assigncount_max, frequency width(1) 
graph export "fig_histogram.pdf", replace as(pdf)


***** Table: Summary statistics *****
gen hum_ss = humanized
label var hum_ss "Pr(Humanized)"
gen art_ss = assigncount
label var art_ss "Articles/homicide"
gen cnt_ss = assigncount
label var cnt_ss "Homicides"
table (vicage_c) (vicsex vicrace_simple) if vicage_c<5 & vicsex<2,  ///
zerocounts nototals statistic(mean hum_ss) statistic(mean art_ss) statistic(count cnt_ss) nformat(%9.2f mean) 
collect title "Summary statistics by age, sex, and race \label{table:summary}"
collect style header result, level(hide)
collect style header vicsex vicrace_simple vicage_c, title(hide)
collect export tab_summary.tex, replace tableonly






*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~ Control sets & sample selection ~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

***** CONTROLS *****

*No interactions, no age
local controls_dem_nia 	i.vicrace i.vicsex  i.offrace i.offsex

*Demographic
local controls_dem 		i.vicage_c##i.vicrace##i.vicsex i.offage_c##i.offrace##i.offsex

*Demographic + Circumstantial
local controls_cir		i.weapon i.relationship i.circumstance i.incident i.agentype i.multiplevic i.multipleoff 

*Demographic + Circumstantial + Geography/Time
local controls_geo 		i.cntyfips  i.year#i.month



***** SAMPLE *****

*Victim race: Black and white
*Victim age: 0-69
*Victim sex: Male or female (not unknown)
keep if vicrace<2 & vicage_c<5 & vicsex<2	







*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~ Main Results: SIMPLE REGRESSION (NO INTERACTIONS/AGE) ~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

*Regressions
reg 	humanizeall `controls_dem_nia', vce(robust)
estimates store lpm_0a
reg 	humanizeall `controls_dem_nia' `controls_cir' `controls_geo', vce(robust)
estimates store lpm_0b
probit 	humanizeall `controls_dem_nia', vce(robust)
estimates store probit_0a
probit 	humanizeall `controls_dem_nia' `controls_cir' `controls_geo', vce(robust)
estimates store probit_0b

*Table
esttab lpm_0a lpm_0b probit_0a probit_0b  using tab_nointeractions.tex, b(2) se(2) ar2 nonotes compress booktabs label replace noabbrev nonumbers star(* 0.1 ** 0.05 *** 0.01) eqlabel(none) ///
mgroups("\normalsize \emph{Lin. Prob. Model}" "\normalsize  \emph{Probit Model}", pattern(1 0 1 0) ///
prefix(\multicolumn{@span}{c}{) suffix(}) ///
span erepeat(\cmidrule(lr){@span})) ///
title(Determinants of Humanizing Coverage of Homicide \label{table:humanizing} \\) ///
keep(1.vicrace 1.vicsex 1.offrace 1.offsex  2.offsex 1.multiplevic 1.multipleoff) ///
coeflabels(1.vicrace "\quad \normalsize Black" 1.vicsex "\quad \normalsize Female" 1.vicrace#1.vicsex "\quad \normalsize Black $\times$ Female" ///
		   1.offrace "\quad \normalsize Black" 1.offsex "\quad \normalsize Female" 1.offrace#1.offsex "\quad \normalsize Black $\times$ Female" ///
		   viccount "\quad \normalsize Victim count" 2.offsex "\quad \normalsize Unsolved" 1.multiplevic "\quad \normalsize Mult. victims" 1.multipleoff "\quad \normalsize Mult. offenders") ///
order(1.vicrace 1.vicsex 1.vicrace#1.vicsex 1.offrace 1.offsex 1.offrace#1.offsex 2.offsex multiplevic multipleoff) ///
refcat(1.vicrace "{\underline{ \normalsize Victim}}" 1.offrace "{ \normalsize \underline{Offender}}" 2.offsex "{\underline{ \normalsize Controls/FE}}", nolabel) ///
substitute(\begin{table}[htbp]\centering \begin{table}[htbp!]\centering\renewcommand{\arraystretch}{1} ///
           {l}{\footnotesize {p{\linewidth}}{\footnotesize ) ///
/// & &\normalsize) ///
mtitles("\normalsize Dem." "\normalsize Full" "\normalsize Dem." "\normalsize Full") ///
addnote("Robust standard errors in parentheses. $^*p<0.1$; $^{**}p<0.05$; $^{***}p<0.01$" ///
 "FE indicates inclusion of fixed effects for the corresponding variable." ///
 "The sample consists of all homicides with a black or white victim under 70 years of age. Sample sizes differ across specifications because, as is well known, perfect predictors can lead to numerical problems in maximum likelihood estimation. As a result, these predictors and their associated observations are automatically dropped in the estimation procedure (as is the default behavior of most statistical packages).") ///
indicate("\normalsize \quad Weapon (FE) = 2.weapon" "\normalsize \quad Circum. (FE) = 2.circumstance" "\normalsize \quad Relation (FE) = 2.relationship"  "\normalsize \quad County (FE) = 1.cntyfips" "\normalsize \quad Time (FE) = 1980.year#1.month", labels("$\times$")) 






	
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~ Main Results: COMPOSITE HUMANIZING SCORE (PROBIT) ~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

*Regression
probit 	humanizeall `controls_dem' `controls_cir' `controls_geo', vce(robust)
local name fig_humanizing

*Margins for plotting
margins vicrace, at(vicage_c=(1(1)4) vicsex=0) asobserved
marginsplot, ///
	recastci(rarea) ///
	plot1op(mcolor(black) lcolor("139 0 0") lwidth(0.4) lpattern(solid) msize(0.6) msymbol(circle)) ci1op(lwidth(0.2) lpattern(solid) color(%10)) ///
	plot2op(mcolor(black) lcolor(edkblue) 	lwidth(0.4) lpattern(shortdash)  msize(0.6) msymbol(square)) ci2op(lwidth(0.2) lpattern(solid) color(%10)) ///
	yline(0, lcolor(black) lwidth(0.2) lstyle(--)) ///
	xlabel(1 "<18" 2 "18-29" 3 "30-49" 4 "50-69", angle(45) labsize(large)) ///
	ylabel(0 "0" 0.2 "0.2" 0.4 "0.4" 0.6 "0.6" 0.8 "0.8" 1 "1",labsize(large)) /// 
	title("Male victim", size(vlarge)) ///
	yscale(range(0 1)) ///
	xtitle("Victim Age", size(vlarge)) ///
	ytitle("Pr(Humanizing coverage)", size(vlarge)) ///
	legend(order(3 "White" 4 "Black") size(large) position(2))
	graph save "`name'_male", replace 
margins vicrace , at(vicage_c=(1(1)4) vicsex=1) asobserved
marginsplot, ///
	recastci(rarea) ///
	plot1op(mcolor(black) lcolor("139 0 0") lwidth(0.4) lpattern(solid) msize(0.6) msymbol(circle)) ci1op(lwidth(0.2) lpattern(solid) color(%10)) ///
	plot2op(mcolor(black) lcolor(edkblue) 	lwidth(0.4) lpattern(shortdash)  msize(0.6) msymbol(square)) ci2op(lwidth(0.2) lpattern(solid) color(%10)) ///
	yline(0, lcolor(black) lwidth(0.2) lstyle(--)) ///
	xlabel(1 "<18" 2 "18-29" 3 "30-49" 4 "50-69", angle(45) labsize(large)) ///
	ylabel(0 "0" 0.2 "0.2" 0.4 "0.4" 0.6 "0.6" 0.8 "0.8" 1 "1" ,labsize(large)) ///
	title("Female victim", size(vlarge)) ///
	yscale(range(0 1)) ///
	legend(order(3 "White" 4 "Black") size(large) position(2)) ///
	xtitle(" ") ///
	ytitle(" ")
	graph save "`name'_female", replace 
graph combine "`name'_male" "`name'_female"
graph display, xsize(10) ysize(5)
graph export "`name'.pdf", replace as(pdf)
graph close

*Margins for testing
collect clear
quietly collect get, tag(model["Male"]): margins r.vicrace , at(vicage_c=(1(1)4) vicsex=0) asobserved
quietly collect get, tag(model["Female"]): margins r.vicrace , at(vicage_c=(1(1)4) vicsex=1) asobserved
collect levelsof model
collect title "Testing significance of black-white differential in composite humanizing score \label{table:test1}"
collect style cell result[_r_se], sformat("(%s)")
collect style header result, level(hide)
collect layout (colname#result[_r_b _r_se]) (model)
collect stars _r_p 0.01 "***" 0.05 "**" 0.1 "*", attach(_r_b)
collect style cell result[_r_b _r_se], nformat(%9.2f) halign(center)
collect export tab_margins1.tex, replace tableonly





*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~ Main Results: EXTENSIVE HUMANIZING SCORE (POISSON) ~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

*Regressions
ppmlhdfe assigncount `controls_dem' `controls_cir' `controls_geo', vce(robust)
local name fig_count

*Margins for plotting
margins vicrace , at(vicage_c=(1(1)4) vicsex=0) asobserved 
marginsplot, ///
	recastci(rarea) ///
	plot1op(mcolor(black) lcolor("139 0 0") lwidth(0.4) lpattern(solid) msize(0.6) msymbol(circle)) ci1op(lwidth(0.2) lpattern(solid) color(%10)) ///
	plot2op(mcolor(black) lcolor(edkblue) 	lwidth(0.4) lpattern(shortdash)  msize(0.6) msymbol(square)) ci2op(lwidth(0.2) lpattern(solid) color(%10)) ///
	yline(0, lcolor(black) lwidth(0.2) lstyle(--)) ///
	xlabel(1 "<18" 2 "18-29" 3 "30-49" 4 "50-69", angle(45) labsize(large)) ///
	yscale(range(0 16)) ///
	ylabel(0 "0" 5 "5" 10 "10" 15 "15",labsize(large)) ///
	title("Male victim", size(vlarge)) ///
	xtitle("Victim Age", size(vlarge)) ///
	ytitle("E[# of articles]", size(vlarge)) ///
	legend(order(3 "White" 4 "Black") size(large) position(2))
	graph save "`name'_male", replace
margins vicrace , at(vicage_c=(1(1)4) vicsex=1) asobserved
marginsplot, ///
	recastci(rarea) ///
	plot1op(mcolor(black) lcolor("139 0 0") lwidth(0.4) lpattern(solid) msize(0.6) msymbol(circle)) ci1op(lwidth(0.2) lpattern(solid) color(%10)) ///
	plot2op(mcolor(black) lcolor(edkblue) 	lwidth(0.4) lpattern(shortdash)  msize(0.6) msymbol(square)) ci2op(lwidth(0.2) lpattern(solid) color(%10)) ///
	yline(0, lcolor(black) lwidth(0.2) lstyle(--)) ///
	xlabel(1 "<18" 2 "18-29" 3 "30-49" 4 "50-69", angle(45) labsize(large)) ///
	yscale(range(0 16)) ///
	ylabel(0 "0" 5 "5" 10 "10" 15 "15",labsize(large)) ///
	title("Female victim", size(vlarge)) ///
	xtitle(" ") ///
	ytitle(" ") ///
	legend(order(3 "White" 4 "Black") size(large) position(2))
	graph save "`name'_female", replace
graph combine "`name'_male" "`name'_female"
graph display, xsize(10) ysize(5)
graph export "`name'.pdf", replace as(pdf)
graph close

*Margins for testing
collect clear
quietly collect get, tag(model["Male"]): margins r.vicrace , at(vicage_c=(1(1)4) vicsex=0) asobserved
quietly collect get, tag(model["Female"]): margins r.vicrace , at(vicage_c=(1(1)4) vicsex=1) asobserved
collect levelsof model
collect title "Testing significance of black-white differential in extensive humanizing score \label{table:test2}"
collect style cell result[_r_se], sformat("(%s)")
collect style header result, level(hide)
collect layout (colname#result[_r_b _r_se]) (model)
collect stars _r_p 0.01 "***" 0.05 "**" 0.1 "*", attach(_r_b)
collect style cell result[_r_b _r_se], nformat(%9.2f) halign(center)
collect export tab_margins2.tex, replace tableonly






*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~ Main Results: CONDITIONAL HUMANIZING SCORE (PROBIT) ~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

*Regressions
probit 	humanized `controls_dem' `controls_cir' `controls_geo' i.assigncount if assigncount>0, vce(robust)
local name fig_language

*Margins for plotting
margins vicrace , at(vicage_c=(1(1)4) vicsex=0) asobserved 
marginsplot, ///
	recastci(rarea) ///
	plot1op(mcolor(black) lcolor("139 0 0") lwidth(0.4) lpattern(solid) msize(0.6) msymbol(circle)) ci1op(lwidth(0.2) lpattern(solid) color(%10)) ///
	plot2op(mcolor(black) lcolor(edkblue) 	lwidth(0.4) lpattern(shortdash)  msize(0.6) msymbol(square)) ci2op(lwidth(0.2) lpattern(solid) color(%10)) ///
	yline(0, lcolor(black) lwidth(0.2) lstyle(--)) ///
	xlabel(1 "<18" 2 "18-29" 3 "30-49" 4 "50-69", angle(45) labsize(large)) ///
	yscale(range(0 1.1)) ///
	ylabel(0 "0" 0.2 "0.2" 0.4 "0.4" 0.6 "0.6" 0.8 "0.8" 1 "1",labsize(large)) /// 
	title("Male victim", size(vlarge)) ///
	xtitle("Victim Age", size(vlarge)) ///
	ytitle("Pr(Humanizing language|Coverage)", size(vlarge)) ///
	legend(order(3 "White" 4 "Black") size(large) position(2))
	graph save "`name'_male", replace
margins vicrace , at(vicage_c=(1(1)4) vicsex=1) asobserved
marginsplot, ///
	recastci(rarea) ///
	plot1op(mcolor(black) lcolor("139 0 0") lwidth(0.4) lpattern(solid) msize(0.6) msymbol(circle)) ci1op(lwidth(0.2) lpattern(solid) color(%10)) ///
	plot2op(mcolor(black) lcolor(edkblue) 	lwidth(0.4) lpattern(shortdash)  msize(0.6) msymbol(square)) ci2op(lwidth(0.2) lpattern(solid) color(%10)) ///
	yline(0, lcolor(black) lwidth(0.2) lstyle(--)) ///
	xlabel(1 "<18" 2 "18-29" 3 "30-49" 4 "50-69", angle(45) labsize(large)) ///
	yscale(range(0 1.1)) ///
	ylabel(0 "0" 0.2 "0.2" 0.4 "0.4" 0.6 "0.6" 0.8 "0.8" 1 "1",labsize(large)) /// 
	title("Female victim", size(vlarge)) ///
	xtitle(" ") ///
	ytitle(" ") ///
	legend(order(3 "White" 4 "Black") size(large) position(2))
	graph save "`name'_female", replace
graph combine "`name'_male" "`name'_female"
graph display, xsize(10) ysize(5)
graph export "`name'.pdf", replace as(pdf)
graph close
	
*Margins for testing
collect clear
quietly collect get, tag(model["Male"]): margins r.vicrace , at(vicage_c=(1(1)4) vicsex=0) asobserved
quietly collect get, tag(model["Female"]): margins r.vicrace , at(vicage_c=(1(1)4) vicsex=1) asobserved
collect levelsof model
collect title "Testing significance of black-white differential in conditional humanizing score \label{table:test3}"
collect style cell result[_r_se], sformat("(%s)")
collect style header result, level(hide)
collect layout (colname#result[_r_b _r_se]) (model)
collect stars _r_p 0.01 "***" 0.05 "**" 0.1 "*", attach(_r_b)
collect style cell result[_r_b _r_se], nformat(%9.2f) halign(center)
collect export tab_margins3.tex, replace tableonly





*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~ Robustness: CONTRAST PLOTS ~~~~~~~~~~~~~~~~~~~~~~~~~*
*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

***** Dropping 1984 *****

*Regression
probit 	humanizeall `controls_dem' `controls_cir' `controls_geo' if year<1984, vce(robust)
local name fig_humanizing_no84

*Margins for plotting
margins vicrace, at(vicage_c=(1(1)4) vicsex=0) asobserved
marginsplot, ///
	recastci(rarea) ///
	plot1op(mcolor(black) lcolor("139 0 0") lwidth(0.4) lpattern(solid) msize(0.6) msymbol(circle)) ci1op(lwidth(0.2) lpattern(solid) color(%10)) ///
	plot2op(mcolor(black) lcolor(edkblue) 	lwidth(0.4) lpattern(shortdash)  msize(0.6) msymbol(square)) ci2op(lwidth(0.2) lpattern(solid) color(%10)) ///
	yline(0, lcolor(black) lwidth(0.2) lstyle(--)) ///
	xlabel(1 "<18" 2 "18-29" 3 "30-49" 4 "50-69", angle(45) labsize(large)) ///
	ylabel(0 "0" 0.2 "0.2" 0.4 "0.4" 0.6 "0.6" 0.8 "0.8" 1 "1",labsize(large)) /// 
	title("Male victim", size(vlarge)) ///
	yscale(range(0 1)) ///
	xtitle("Victim Age", size(vlarge)) ///
	ytitle("Pr(Humanizing coverage)", size(vlarge)) ///
	legend(order(3 "White" 4 "Black") size(large) position(2))
	graph save "`name'_male", replace 
margins vicrace, at(vicage_c=(1(1)4) vicsex=1) asobserved
marginsplot, ///
	recastci(rarea) ///
	plot1op(mcolor(black) lcolor("139 0 0") lwidth(0.4) lpattern(solid) msize(0.6) msymbol(circle)) ci1op(lwidth(0.2) lpattern(solid) color(%10)) ///
	plot2op(mcolor(black) lcolor(edkblue) 	lwidth(0.4) lpattern(shortdash)  msize(0.6) msymbol(square)) ci2op(lwidth(0.2) lpattern(solid) color(%10)) ///
	yline(0, lcolor(black) lwidth(0.2) lstyle(--)) ///
	xlabel(1 "<18" 2 "18-29" 3 "30-49" 4 "50-69", angle(45) labsize(large)) ///
	ylabel(0 "0" 0.2 "0.2" 0.4 "0.4" 0.6 "0.6" 0.8 "0.8" 1 "1" ,labsize(large)) ///
	title("Female victim", size(vlarge)) ///
	yscale(range(0 1)) ///
	legend(order(3 "White" 4 "Black") size(large) position(2)) ///
	xtitle(" ") ///
	ytitle(" ")
	graph save "`name'_female", replace 
graph combine "`name'_male" "`name'_female", title("Robustness: Excluding 1984", size(large))
graph display, xsize(10) ysize(5)
graph export "`name'.pdf", replace as(pdf)
graph close	

	
***** Agency FE *****

*Alternative FE
local controls_geoalt i.agency  i.year#i.month

*Regression
probit 	humanizeall `controls_dem' `controls_cir' `controls_geoalt', vce(robust)
local name fig_humanizing_agencyfe

*Margins for plotting
margins vicrace, at(vicage_c=(1(1)4) vicsex=0) asobserved
marginsplot, ///
	recastci(rarea) ///
	plot1op(mcolor(black) lcolor("139 0 0") lwidth(0.4) lpattern(solid) msize(0.6) msymbol(circle)) ci1op(lwidth(0.2) lpattern(solid) color(%10)) ///
	plot2op(mcolor(black) lcolor(edkblue) 	lwidth(0.4) lpattern(shortdash)  msize(0.6) msymbol(square)) ci2op(lwidth(0.2) lpattern(solid) color(%10)) ///
	yline(0, lcolor(black) lwidth(0.2) lstyle(--)) ///
	xlabel(1 "<18" 2 "18-29" 3 "30-49" 4 "50-69", angle(45) labsize(large)) ///
	ylabel(0 "0" 0.2 "0.2" 0.4 "0.4" 0.6 "0.6" 0.8 "0.8" 1 "1",labsize(large)) /// 
	title("Male victim", size(vlarge)) ///
	yscale(range(0 1)) ///
	xtitle("Victim Age", size(vlarge)) ///
	ytitle("Pr(Humanizing coverage)", size(vlarge)) ///
	legend(order(3 "White" 4 "Black") size(large) position(2))
	graph save "`name'_male", replace 
margins vicrace, at(vicage_c=(1(1)4) vicsex=1) asobserved
marginsplot, ///
	recastci(rarea) ///
	plot1op(mcolor(black) lcolor("139 0 0") lwidth(0.4) lpattern(solid) msize(0.6) msymbol(circle)) ci1op(lwidth(0.2) lpattern(solid) color(%10)) ///
	plot2op(mcolor(black) lcolor(edkblue) 	lwidth(0.4) lpattern(shortdash)  msize(0.6) msymbol(square)) ci2op(lwidth(0.2) lpattern(solid) color(%10)) ///
	yline(0, lcolor(black) lwidth(0.2) lstyle(--)) ///
	xlabel(1 "<18" 2 "18-29" 3 "30-49" 4 "50-69", angle(45) labsize(large)) ///
	ylabel(0 "0" 0.2 "0.2" 0.4 "0.4" 0.6 "0.6" 0.8 "0.8" 1 "1" ,labsize(large)) ///
	title("Female victim", size(vlarge)) ///
	yscale(range(0 1)) ///
	legend(order(3 "White" 4 "Black") size(large) position(2)) ///
	xtitle(" ") ///
	ytitle(" ")
	graph save "`name'_female", replace 
graph combine "`name'_male" "`name'_female", title("Robustness: Agency fixed effects", size(large))
graph display, xsize(10) ysize(5)
graph export "`name'.pdf", replace as(pdf)
graph close		
	

***** Excl. article outliers *****

*Regression
probit 	humanizeall `controls_dem' `controls_cir' `controls_geo' if assigncount<maxval, vce(robust)
local name fig_humanizing_trim

*Margins for plotting
margins vicrace, at(vicage_c=(1(1)4) vicsex=0) asobserved
marginsplot, ///
	recastci(rarea) ///
	plot1op(mcolor(black) lcolor("139 0 0") lwidth(0.4) lpattern(solid) msize(0.6) msymbol(circle)) ci1op(lwidth(0.2) lpattern(solid) color(%10)) ///
	plot2op(mcolor(black) lcolor(edkblue) 	lwidth(0.4) lpattern(shortdash)  msize(0.6) msymbol(square)) ci2op(lwidth(0.2) lpattern(solid) color(%10)) ///
	yline(0, lcolor(black) lwidth(0.2) lstyle(--)) ///
	xlabel(1 "<18" 2 "18-29" 3 "30-49" 4 "50-69", angle(45) labsize(large)) ///
	ylabel(0 "0" 0.2 "0.2" 0.4 "0.4" 0.6 "0.6" 0.8 "0.8" 1 "1",labsize(large)) /// 
	title("Male victim", size(vlarge)) ///
	yscale(range(0 1)) ///
	xtitle("Victim Age", size(vlarge)) ///
	ytitle("Pr(Humanizing coverage)", size(vlarge)) ///
	legend(order(3 "White" 4 "Black") size(large) position(2))
	graph save "`name'_male", replace 
margins vicrace, at(vicage_c=(1(1)4) vicsex=1) asobserved
marginsplot, ///
	recastci(rarea) ///
	plot1op(mcolor(black) lcolor("139 0 0") lwidth(0.4) lpattern(solid) msize(0.6) msymbol(circle)) ci1op(lwidth(0.2) lpattern(solid) color(%10)) ///
	plot2op(mcolor(black) lcolor(edkblue) 	lwidth(0.4) lpattern(shortdash)  msize(0.6) msymbol(square)) ci2op(lwidth(0.2) lpattern(solid) color(%10)) ///
	yline(0, lcolor(black) lwidth(0.2) lstyle(--)) ///
	xlabel(1 "<18" 2 "18-29" 3 "30-49" 4 "50-69", angle(45) labsize(large)) ///
	ylabel(0 "0" 0.2 "0.2" 0.4 "0.4" 0.6 "0.6" 0.8 "0.8" 1 "1" ,labsize(large)) ///
	title("Female victim", size(vlarge)) ///
	yscale(range(0 1)) ///
	legend(order(3 "White" 4 "Black") size(large) position(2)) ///
	xtitle(" ") ///
	ytitle(" ")
	graph save "`name'_female", replace 
graph combine "`name'_male" "`name'_female", title("Robustness: Excluding high-coverage homicides", size(large))
graph display, xsize(10) ysize(5)
graph export "`name'.pdf", replace as(pdf)
graph close		



