*******************************************************************************
***************			READ ME	  	  *****************************
*******************************************************************************

NOTE 21. DECEMBER 2021: se opdatering om reinfektioner under 01_Noegle_tal.
Bemærk at reinfektioner som udgangspunkt ikke inkluderes i opgørelser med mindre det angives.



Her finder du en beskrivelse af indeholdet af det regionale dashboard, herunder 
beskrivelser af tabellerne med tilhørende variabelnavne (søjlenavne).


Forkortelser:
_______________________________________________________________________________
TCD = Test Center Danmark (Statens Serum Institut)
KMA = Klinisk Mikrobiologisk Afdeling (hospitaler)


Generel struktur:
_______________________________________________________________________________
Rækkerne i filerne er som udgangspunkt opdelt efter relevante 
parametre, eksempelvis aldersgruppering eller tidsopdeling. 
Der opdeles generelt efter variablen i første søjle. 
Enkelte tabeller kan have rækker, som afviger fra dette mønster. 
I disse tabeller specificeres dette i "Noter" under tabellens 
variabelbeskrivelse. Filerne er kommaseparerede.

Definitioner: 
_______________________________________________________________________________
Definitioner der benyttes i dette og datafilerne er beskrevet på SSI's COVID-19-
side under Datakilder og definitioner. 
https://covid19.ssi.dk/datakilder-og-definitioner
Den 8. december 2021 ændredes definitionen af "seneste 7 dage". 

Opdateringer:
_______________________________________________________________________________
Filerne bliver opdateret hver dag. I den forbindelse kan tidsserier også 
ændre sig tilbage i tiden, hvis nyere data foreligger. Derfor anbefales det 
altid at benytte det senest tilgængelige data og for så vidt muligt, 
ikke at gemme filer og lave tidsserier på basis af gamle filer.


Typer af tests:
_________________________________________________________________________
Filerne baseres primært på PCR-opgørelser, medmindre andet er angivet. 
PCR-tests og antigentests bruges til hhv. at bekræfte eller mistænke covid-19-
smitte under et aktivt sygdomsforløb. 
Begreberne "PCR-prøver" og "PCR-test" betyder det samme og refererer til
antallet af podninger. Det samme gælder for begreberne "antigentest" og 
"antigenprøver".
Data indeholder ikke serologitest, som er den test, der udføres, 
når man skal undersøge, om raske mennesker tidligere har haft covid-19.

Filerne:
_______________________________________________________________________________

01_Noegle_tal
-------------------------------------------------------------------------------
Note: fra den 21. december 2021 er nogle variable relateret til reinfektioner blevet tilføjet.
Fra den 21. december 2021 og frem inkluderer tallene for dødsfald og nyindlagte også personer med reinfektioner.

Denne fil indeholder dagens nøgletal for covid-19-epidemien, som er de tal der også vises øverst på det regionale dashboard.
Både totaler og ændringer siden sidste opdatering er inkluderet. Tallene er opdelt på region og køn
NA i "Region"-variablen repræsenterer danske personer uden en tilknyttet bopælsregion. 


Dato						: Opgørelsesdato for nøgletal
Region						: Bopælsregion (region man boede i ved prøvetagning)
Køn						: Mænd og kvinder
Bekræftede tilfælde i alt			: Bekræftede covid-19-tilfælde fra både førstegangsinfektioner og reinfektioner ("Bekræftede tilfælde i alt" = "Bekræftede tilfælde" + "Reinfektioner")
Bekræftede tilfælde				: Bekræftede covid-19-tilfælde (kun førstegangsinfektioner)
Reinfektioner					: Bekræftede covid-19-tilfælde (kun reinfektioner)
Døde						: Dødsfald med covid-19 i alt
Indlæggelser					: Nyindlagte med covid-19 i alt
Testede personer				: PCR-testede personer i alt
Antallet af prøver				: Antal PCR-tests i alt
Indlæggelser i dag (psykiatri)	: Indlagte med covid-19 på opgørelsesdatoen (kun i psykiatriske afdelinger)
Ændring i antal bekræftede tilfælde i alt	: Nye registrerede covid-19-tilfælde siden sidste opdatering fra både førstegangsinfektioner og reinfektioner
Ændring i antal bekræftede tilfælde		: Nye registrerede covid-19-tilfælde siden sidste opdatering (kun førstegangsinfektioner)
Ændring i antal reinfektioner			: Nye registrerede covid-19-tilfælde siden sidste opdatering (kun reinfektioner)
Ændring i antal døde				: Nye registrerede dødsfald med covid-19 siden sidste opdatering
Ændring i antal indlagte			: Nye indlæggelser med covid-19 siden sidste opdatering
Ændring i antallet af testede personer		: Antallet af PCR-testede personer siden sidste opdatering
Ændring i antallet af PCR prøver		: Antal nye PCR-tests siden sidste opdatering
Ændring i antallet af Antigen prøver		: Antal nye antigen-tests siden sidste opdatering
Ændring i indlæggelser (psykiatri) siden sidste opdatering	: Ændring i antal indlagte med covid-19 siden sidste opdatering (kun i psykiatriske afdelinger)


02_Hospitalsbelaegning
-------------------------------------------------------------------------------
Dette er den daglige opgørelse for den covid-19-relaterede hospitalsbelægning,
opgjort pr. region.
Bemærk, at der er forskel mellem antallet af personer indlagt siden i går, opgjort i 
filen: <01_Noegletal>, og ændringen i antallet af indlagte. Ændringen er den reelle 
ændring observeret, hvor antallet af indlæggelser siden i går udgør det antal nye 
indlæggelser siden i går.

Dato					: Dato for prøvetagning og eventuel indlæggelsesdag
Region					: Indlæggelsesregion
Indlagte				: Antallet af personer indlagt
Heraf indlagte på 
	intensiv i
	respirator			: Antallet af personer der er indlagt på intensiv og ligger i respirator 

Heraf indlagte på 		 	 
	intensiv			: Antallet af personer indlagt på intensiv
Ændring i antal indlagte		: Ændringen i antallet af indlagte siden i går	
Ændring i antal indlagte på intensiv	: Ændringen i antallet af indlagte på intensiv siden i går
Ændring i antal indlagte i respirator 	: Ændringen i antallet af indlagte på intensiv i respirator
				  siden i går

03_bekraeftede tilfaelde_doede_indlagte_pr_dag_pr_koen
-------------------------------------------------------------------------------
Dette er en opgørelse over antallet af bekræftede tilfælde og døde pr. dag
fordelt på bopælsregioner og køn.

Region					: Bopælsregion (region man boede i ved prøvetagning)		
Dato					: Dato for prøvetagning og evntuel indlæggelsesdag
Køn						: Mænd og kvinder 
Bekræftede tilfælde			: Antallet af bekræftede tilælde
Døde					: Antallet af døde
Indlæggelser				: Antallet af indlæggelser
Kumuleret antal døde			: Alle dødsfald fra pandemiens start til den aktuelle dag
Kumuleret antal 		
	bekræftede tilfælde		: Alle bekræftede tilfælde fra pandemiens start til den aktuelle dag
Kumuleret antal 		 	
	indlæggelser			: Alle indlæggelser fra pandemiens start til den aktuelle dag

04_indlagte_pr_alders_grp_pr_region
-------------------------------------------------------------------------------
Dette er en opgørelse over antallet af indlagte pr. aldersgruppe pr. region.
Bemærk at blanke felter under variablen <Region> udgør de danskere,
som ikke har en tildelt bopælsregion.

Region					: Bopælsregion (region man boede i ved indlæggelsen)
Aldersgruppe				: Den aldersgruppe en person tilhørte ved
						  prøvetagningen
Indlæggelser				: Kumulerede antal indlæggelser

05_bekraeftede_tilfaelde_doede_pr_region_pr_alders_grp
-------------------------------------------------------------------------------
Dette er en opgørelse over antallet af bekræftede tilfælde og døde pr. region 
og pr. aldergruppe. Bemærk at blanke felter under variablen <Region> 
udgør de danskere, som ikke har en tildelt bopælsregion.

Region					: Bopælsregion (region man boede i ved prøvetagningen)
Aldersgruppe				: Den aldersgruppe en person tilhørte ved
						  prøvetagningen
Bekræftede tilfælde			: Kumulerede antal bekræftede tilælde
Døde					: Kumulerede antal døde

06_nye_indlaeggelser_pr_region_pr_dag
-------------------------------------------------------------------------------
Dette er en opgørelse over antallet af nye indlæggelser pr. region pr. dag.

Region					: Bopælsregion (region man boede i ved indlæggelsen)		
Dato					: Dato for indlæggelsen
Indlæggelser				: Antallet af indlæggelser på en given dag i en given
						  region
07_antal_doede_pr_dag_pr_region
-------------------------------------------------------------------------------
Dette er en opgørelse over antallet af døde pr. region pr. dag. 
Bemærk at blanke felter under variablen <Region> udgør de danskere, 
som ikke har en tildelt bopælsregion.

Region					: Bopælsregion (region man boede i ved prøvetagningen)
Dato					: Dato for dødsfald registreret Dødsårsagsregisteret
Kumulerede antal døde			: Kumulerede antal døde siden pandemiens start

08_bekraeftede_tilfaelde_pr_dag_pr_regions
-------------------------------------------------------------------------------
Dette er en opgørelse over antallet af bekræftede tilfælde pr. region pr.
dag siden pandemiens start.

Region					: Bopælsregion (region man boede i ved prøvetagningen)
Dato					: Dato for dødsfald registreret i Dødsårsagsregisteret
Bekræftede tilfælde			: Antallet af bekræftede tilælde

09_bekraeftede_tilfaelde_pr_test_region_pr_uge
-------------------------------------------------------------------------------
Dette er en opgørelse af antallet af bekræftede tilfælde pr. uge pr. region
opgjort på den region, hvor PCR-testen blev taget.

Uge					: Den uge hvor testen blev taget
TCD Region Hovedstaden			: Alle positive PCR-test i region Hovedstaden testet af TCD
Region Hovedstaden			: Alle positive PCR-test i region Hovedstaden testet af KMA
TCD Region Sjælland			: Alle positive PCR-test i region Sjælland testet af TCD
Region Sjælland				: Alle positive PCR-test i region Sjælland testet af KMA
TCD Region Nordjylland			: Alle positive PCR-test i region Nordjylland testet af TCD
Region Nordjylland			: Alle positive PCR-test i region Nordjylland testet af KMA
TCD Region Syddanmark			: Alle positive PCR-test i region Syddanmark testet af TCD
Region Syddanmark			: Alle positive PCR-test i region Syddanmark testet af KMA
TCD_Region Midtjylland			: Alle positive PCR-test i region Midtjylland testet af TCD
Region Midtjylland			: Alle positive PCR-test i region Midtjylland testet af KMA
Sundhedsspor				: Alle positive PCR-test registreret af KMA
Samfundsspor				: Alle positive PCR-test registreret af TCD
Total					: Det samlede antal bekræftede positive PCR-test.

10_testede_pr_uge_pr_samfundsspor_opgjort_paa_bopaelsregion
-------------------------------------------------------------------------------
Dette er en opgørelse af antallet af PCR-testede pr. uge pr. region opgjort på
bopælsregion.

Uge				: Den uge hvor PCR-testen blev foretaget.
Region Hovedstaden		: Alle PCR-testede i region Hovedstaden testet af KMA
Region Sjælland			: Alle PCR-testede i region Sjælland testet af KMA
Region Nordjylland		: Alle PCR-testede i region Nordjylland testet af KMA
Region Syddanmark		: Alle PCR-testede i region Syddanmark testet af KMA
Region Midtjylland		: Alle PCR-testede i region Midtjylland testet af KMA

11_noegletal_pr_region_pr_aldersgruppe
-------------------------------------------------------------------------------
Dette er en opgørelse af antallet af bekræftede tilfælde, døde, indlagte, herunder på intensiv 
afdeling opgjort pr. region og pr. aldersgruppe.


Region					: Bopælsregion (region man boede i ved prøvetagningen)
Aldersgruppe				: Den  aldersgruppe en person tilhørte ved
						  prøvetagningen
Bekræftede tilfælde			: Antallet af bekræftede tilfælde
Døde					: Antallet af døde
Indlagte på intensiv
	afdeling			: Antallet af patienter indlagt på intensiv afdeling
Indlæggelser				: Antallet af indlagte

12_noegletal_pr_region_pr_aldersgruppe_de_seneste_7_dage
-------------------------------------------------------------------------------
Dette er en opgørelse af antallet af bekræftede tilfælde, døde, indlagte,
herunder på intensiv afdeling opgjort pr. region og pr. aldersgruppe de seneste 7 dage.

Region					: Bopælsregion (region man boede i ved prøvetagningen)
Aldersgruppe				: Den aldersgruppe en person tilhørte ved
						  prøvetagningen
Bekræftede tilfælde			: Antallet af bekræftede tilfælde
Døde					: Antallet af døde
Indlagte på intensiv
	afdeling			: Antallet af patienter indlagt på intensiv afdeling
Indlæggelser				: Antallet af indlagte

13_regionale_kort
-------------------------------------------------------------------------------
Dette er en opgørelse af antallet af bekræftede tilfælde, incidensen, bekræftede
tilfælde de seneste 7 dage, incidensen de seneste 7 dage, PCR-testede personer,
PCR-testede pr. 100.000 borgere, PCR-testede de seneste 7 dage, PCR-testede pr. 
100.000 borgere de seneste 7 dage, samt positivprocenten de seneste 7 dage, opgjort pr. region. 
Læs mere om opgørelsesmetoden for de seneste 7 dage under "Definitioner og datakilder" på 
https://covid19.ssi.dk/datakilder-og-definitioner.

Region					: Bopælsregion (region man boede i ved prøvetagningen)
Bekræftede tilfælde			: Antallet af bekræftede tilfælde siden pandemiens start
Incidensen				: Antallet af bekræftede tilfælde pr. 100.000 borgere
Bekræftede tilfælde
	de seneste 7 dage		: Antallet af bekræftede tilfælde de seneste 7 dage
Incidensen de seneste
	7 dage				: Antallet af bekræftede tilfælde pr. 100.000 borgere de seneste 7 dage
Testede					: Antallet af PCR-testede personer siden pandemiens start
Testede pr. 100.000 borgere		: Antallet af PCR-testede pr. 100.000 borgere
Testede de seneste 		
	7 dage				: Antallet af PCR-testede personer de seneste 7 dage 
Testede pr. 100.000 borgere 
	de seneste 7 dage		: Antallet af PCR-testede pr. 100.000 borgere de seneste 7 dage
Positivprocent 			 
	de seneste 7 dage		: (antal covid-19-bekræftede personer/antallet af PCR-testede personer) *100 for de seneste 7 dage

14_Testede_pr_test_region_pr_uge
-------------------------------------------------------------------------------
Dette er en opgørelse af antallet af PCR-testede pr. region pr. uge, samt det samlede antal PCR-testede 
i hhv. samfundssporet (TCD) og sundhedssporet (KMA).

Uge					: Den uge hvor PCR-testen blev taget
TCD Region Hovedstaden			: Alle PCR-test i region Hovedstaden foretaget af TCD
Region Hovedstaden			: Alle PCR-test i region Hovedstaden foretaget af KMA
TCD Region Sjælland			: Alle PCR-test i region Sjælland foretaget af TCD
Region Sjælland				: Alle PCR-test i region Sjælland foretaget af KMA
TCD Region Nordjylland			: Alle PCR-test i region Nordjylland foretaget af TCD
Region Nordjylland			: Alle PCR-test i region Nordjylland foretaget af KMA
TCD Region Syddanmark			: Alle PCR-test i region Syddanmark foretaget af TCD
Region Syddanmark			: Alle PCR-test i region Syddanmark foretaget af KMA
TCD_Region Midtjylland			: Alle PCR-test i region Midtjylland foretaget af TCD
Region Midtjylland			: Alle PCR-test i region Midtjylland foretaget af KMA
Sundhedsspor				: Alle PCR-test registreret af KMA
Samfundsspor				: Alle PCR-test registreret af TCD
Total					: Det samlede antal PCR-test

15_Indlagte_pr_region_pr_dag
-------------------------------------------------------------------------------
På datoer fra d. 13. december 2021 og frem tæller reinficerede personer med i hospitalsbelægningen.
Fra d. 12. december 2021 og tilbage tæller reinficerede personer ikke med.

Dette er en opgørelse af antallet af indlagte på en given dag fordelt på 
regioner. 

Dato					: Dato for indlæggelse 
Region					: Regionen hvor patienten er indlagt
Indlagte				: Antallet af indlæggelser
Indlagte på intensiv			: Antallet af indlæggelser på intensivafdelinger
Indlagte i respirator			: Antallet af indlæggelser i respirator
Indlagte i psykiatrien			: Antallet af indlæggelser på psykiatriske afdelinger

16_pcr_og_antigen_test_pr_region
-------------------------------------------------------------------------------
Uge		: Årstal og ugenummer
Region	: Regionsnavn
Prøver		: Antal prøver foretaget
Metode		: Prøvetagningsmetode (Antigen eller PCR)

Note: Der forekommer rækker uden Regionsnavn. Det kan fx forekomme, hvis testede personer har et CPR-nummer, men ikke er bosat i Danmark. 

17_koen_uge_testede_positive_nyindlagte
-------------------------------------------------------------------------------
Dette er en opgørelse over testede, positive og nyindlagte, angivet i antal personer pr. 100.000 personer.
Alle tre nøgletal er angivet på ugebasis og opdelt efter køn.

Uge				: Den uge testen er blevet taget eller nyindlæggelsen er forekommet
Køn				: M = Mænd, K = Kvinder
Testede pr. 100.000 borgere	: Antal testede personer pr. 100.000 borgere i den pågældende uge
Positive pr. 100.000 borgere	: Antal personer med positive tests pr. 100.000 borgere i den pågældende uge
Nyindlagte pr. 100.000 borgere	: Antal nyindlagte personer pr. 100.000 borgere i den pågældende uge

18_fnkt_alder_uge_testede_positive_nyindlagte | OBS 5. oktober 2021: Se note i bunden af denne fil
-------------------------------------------------------------------------------
Dette er en opgørelse over testede, positive og nyindlagte, angivet i antal personer pr. 100.000 personer.
Alle tre nøgletal er angivet på ugebasis og opdelt efter funktionelle aldersgrupper.

Uge				: Den uge testen er blevet taget eller nyindlæggelsen er forekommet
Aldersgruppe			: Aldersgruppe
Testede pr. 100.000 borgere	: Antal testede personer pr. 100.000 borgere i den pågældende uge
Positive pr. 100.000 borgere	: Antal personer med positive tests pr. 100.000 borgere i den pågældende uge
Nyindlagte pr. 100.000 borgere	: Antal nyindlagte personer pr. 100.000 borgere i den pågældende uge

19_indlagte_pr_fnkt_alder_pr_region
-------------------------------------------------------------------------------
Dette er en opgørelse over antallet af indlagte pr. aldersgruppe pr. region.
Bemærk at blanke felter under variablen <Region> udgør de danskere,
som ikke har en tildelt bopælsregion.

Region					: Bopælsregion (region man boede i ved indlæggelsen)
Aldersgruppe				: Den aldersgruppe en person tilhørte ved
						  prøvetagningen
Indlæggelser				: Kumulerede antal indlæggelser

20_bekraeftede_tilfaelde_pr_region_pr_fnkt_alder
-------------------------------------------------------------------------------
Dette er en opgørelse over antallet af bekræftede tilfælde pr. region 
og pr. aldergruppe. Bemærk at blanke felter under variablen <Region> 
udgør de danskere, som ikke har en tildelt bopælsregion.

Region					: Bopælsregion (region man boede i ved prøvetagningen)
Aldersgruppe				: Den aldersgruppe en person tilhørte ved
						  prøvetagningen
Bekræftede tilfælde			: Kumulerede antal bekræftede tilælde

21_noegletal_pr_region_pr_fnkt_alder
-------------------------------------------------------------------------------
Dette er en opgørelse af antallet af bekræftede tilfælde, døde, indlagte, herunder på intensiv 
afdeling opgjort pr. region og pr. aldersgruppe.


Region					: Bopælsregion (region man boede i ved prøvetagningen)
Aldersgruppe				: Den  aldersgruppe en person tilhørte ved
						  prøvetagningen
Bekræftede tilfælde			: Antallet af bekræftede tilfælde
Døde					: Antallet af døde
Indlagte på intensiv
	afdeling			: Antallet af patienter indlagt på intensiv afdeling
Indlæggelser				: Antallet af indlagte

22_noegletal_pr_region_pr_fnkt_alder_de_seneste_7_dage
-------------------------------------------------------------------------------
Dette er en opgørelse af antallet af bekræftede tilfælde, døde, indlagte,
herunder på intensiv afdeling opgjort pr. region og pr. aldersgruppe de seneste 7 dage.

Region					: Bopælsregion (region man boede i ved prøvetagningen)
Aldersgruppe				: Den aldersgruppe en person tilhørte ved
						  prøvetagningen
Bekræftede tilfælde			: Antallet af bekræftede tilfælde
Døde					: Antallet af døde
Indlagte på intensiv
	afdeling			: Antallet af patienter indlagt på intensiv afdeling
Indlæggelser				: Antallet af indlagte

23_reinfektioner_uge_region
-------------------------------------------------------------------------------
Uge								: Den uge testen er blevet taget, hvor reinfektionen er forkommet. 
Region								: Bopælsregion (region man boede i ved prøvetagningen)
Infected							: Antallet af infektioner
Reinfektioner pr. 100.000 borgere				: Antal bekræftede reinfektioner pr. 100.000 borgere i den pågældende uge
Type								: Type af infektion (bekræftet tilfælde eller reinfektion)

24_reinfektioner_daglig_region
-------------------------------------------------------------------------------
Prøvedato							: Den dato testen er blevet taget, hvor reinfektionen er forkommet. 
Region								: Bopælsregion (region man boede i ved prøvetagningen)
Antal reinfektioner					: Antallet af bekræftede reinfektioner
Antal borgere						: Antal borgere den pågældende måned

Definition af reinfektion:
Når en person modtager en positiv PCR-prøve tæller vedkommende som et bekræftet tilfælde. 
Hvis samme person 61 dage eller mere efter prøvetagningsdatoen for denne prøve modtager endnu en positiv PCR-prøve, 
vil vedkommende tælle med igen som en reinfektion. 
61 dage efter denne prøve er det muligt for samme person at blive reinficeret igen 
- bemærk at der ikke er krav om en negativ PCR-test mellem forskellige infektions-/reinfektionforløb, 
og at der ikke er nogen begrænsning på antallet af reinfektionsforløb én person kan have.