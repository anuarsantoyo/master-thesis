*******************************************************************************
***************			READ ME	  	  *****************************
*******************************************************************************

Her finder du en beskrivelse af indeholdet af zip-filen til SSI's testcenter 
dashboard, herunder beskrivelser af tabellerne med tilhørende variabelnavne 
(søjlenavne).


Generel struktur:
_______________________________________________________________________________
Rækkerne i filerne er som udgangspunkt stratificeringer efter relevante 
parametre, eksempelvis aldersgruppering eller tidsopdeling. Filerne er 
comma-separerede.

Definitioner: 
_______________________________________________________________________________
Definitioner der benyttes i dette og datafilerne er beskrevet på SSI's COVID-19-
side under Datakilder og definitioner. 
https://covid19.ssi.dk/datakilder-og-definitioner
Den 8. december 2021 ændredes definitionen af "seneste 7 dage". 

Opdateringer:
_______________________________________________________________________________
Filerne bliver opdateret hver dag og i denne forbindelse kan tidsserier også 
ændrer sig tilbage i tiden, hvis nyere data foreligger. Derfor anbefales det 
altid at benytte det senest tilgængelige data og for så vidt muligt, ikke at 
gemme filer og lave tidsserier på basis af gamle filer.


Datakilder:
_______________________________________________________________________________

- PCR og antigen-tests og testede personer (Dansk Mikrobiologisk database (MiBa))
	- Testcenter koordinater (PCR) kobles via 
	  https://coronasmitte.dk/raad-og-regler/kort-over-covid-19-testcentre 
	  (rest api) via ydernummer
	- Testcenter koordinater (antigen) kobles via SOR-koder fra 
	  SOR-registeret 
	- Data for personer uden dansk CPR-nummer (MiBa)


Diskretionering:
_______________________________________________________________________________

Data diskretioneres, hvis der er færre end 20 testede personer pr. testcenter. 


Data behandling:
_______________________________________________________________________________

Der findes ydernumre (PCR-testcentre), som får forskellige koordinater 
tilknyttet, her vises kun et af disse testcentre i dashboardet. 
 

Filerne:
_______________________________________________________________________________


Fil: 01_pcr_positiv_procent_pr_test_center_de_seneste_7_dage:
------------------------------------------------------------------------------
Ydernummer: 					Testcenter kode 
Testcenter navn: 				Navn på testcenter (evt. by)
Testede personer de seneste 7 dage: 		Antal af PCR-testede unikke personer de seneste estimerede 7 dage
Bekræftede tilfælde de seneste 7 dage: 	Antal af PCR-bekræftede covid tilfælde (unikke personer) de seneste estimerede 7 dage
Positiv procenten de seneste 7 dage: 		Bekræftede tilfælde de seneste 7 dage / Testede pers. de seneste 7 dage * 100 


Fil: 02_antigen_test_pr_test_center_de_seneste_7_dage
------------------------------------------------------------------------------
Antal tests: 					Antal antigen tests foretaget
Testcenter navn: 				Navn på testcenter (bynavn m.m.)
Sor-kode: 					Kode på max 18 karakterer som angivet i Sundhedsvæsenets- Organisationsregister (SOR). Anvendes også til at finde koordinater på hvert testcenter til at placere testcenteret geografisk. 
Længdegrader: 					x-koordinat
Breddegrader: 					y-koordinat




Fil: 04_turist_positiv_procent_de_seneste_7_dage ***
------------------------------------------------------------------------------
Antal tests: 					Antal PCR-tests de seneste estimerede 7 dage
Positive tests: 				Antal PCR-Positive covid-19 tilfælde de seneste estimerede 7 dage
Positiv procent de seneste 7 dage: 		Bekræftede tilfælde de seneste 7 dage / antal tests de seneste 7 dage * 100 



Fil: 05_pcr_testede_personer_pr_test_center_pr_alders_grp_de_seneste_7_dage
------------------------------------------------------------------------------
Ydernummer: 					Testcenter ydernummer kode 
Aldersgruppe: 					0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90+ år
PCR testede personer de seneste 7 dage:	Antal PCR testede personer de seneste estimerede 7 dage



Fil: 06_ag_testede_personer_pr_test_center_pr_alders_gruppe_de_seneste_7_dage
------------------------------------------------------------------------------
SOR-kode: 					Testcenter SOR-kode
Aldersgruppe:					0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90+ år
Antigen testede personer de seneste 7 dage: 	Antal antigen testede personer de seneste estimerede 7 dage



Fil: 07_pcr_testede_personer_pr_test_center_pr_fnkt_alder_de_seneste_7_dage
------------------------------------------------------------------------------
Ydernummer: 					Testcenter ydernummer kode 
Aldersgruppe: 					0-2, 3-5, 6-11, 12-15, 16-19, 20-39, 40-64, 65-79, 80+ år
PCR testede personer de seneste 7 dage:	Antal PCR testede personer de seneste estimerede 7 dage



Fil: 08_ag_testede_personer_pr_test_center_pr_fnkt_alder_de_seneste_7_dage
------------------------------------------------------------------------------
SOR-kode: 					Testcenter SOR-kode
Aldersgruppe:					0-2, 3-5, 6-11, 12-15, 16-19, 20-39, 40-64, 65-79, 80+ år
Antigen testede personer de seneste 7 dage: 	Antal antigen testede personer de seneste estimerede 7 dage


Noter:
_______________________________________________________________________________   
*** Med "turist" menes der her personer uden dansk CPR-nummer, heriblandt turister, vandrende arbejdstagere og asylansøgere. 

