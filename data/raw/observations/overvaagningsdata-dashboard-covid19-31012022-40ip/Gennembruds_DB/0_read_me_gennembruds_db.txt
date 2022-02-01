/*___________________________________________________________________________*/
/*									     */
/*				READ ME					     */
/*___________________________________________________________________________*/

Her finder du en beskrivelse af indeholdet af zip-filen til SSI's Gennembruds 
dashboard, herunder beskrivelser af tabellerne med tilhørende variabelnavne 
(søjlenavne).


Generel struktur:
_______________________________________________________________________________
Rækkerne i filerne er som udgangspunkt stratificeringer efter relevante 
parametre, eksempelvis aldersgruppering eller tidsopdeling. Filerne er 
semicolon-separerede.

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


Filerne:
_______________________________________________________________________________


Fil: 01_indlagte_pr_vaccinationsstatus
------------------------------------------------------------------------------
Denne benyttes til at beregne nøgletal og søjle diagrammet, for indlagte pr. 100.000

Kolonner:
Vaccinationsstatus:					Vaccinationsstatus ved prøvetagning med positivt resultat.
Antal indlagte: 					Antal indlagte
Befolkningstørrelse: 					Antal af personer tilhørende den given vaccinestatus
Indlagte pr. 100.000: 					Antal indlagte divideret med den given befolkningsstørrelse ganget med 100.000.




Fil: 02_bekræftede_tilfælde_pr_vaccinationsstatus
------------------------------------------------------------------------------
Denne benyttes til at beregne nøgletal og søjle diagrammet, for bekræftede tilfælde pr. 100.000

Kolonner:
Vaccinationsstatus:					Vaccinationsstatus ved prøvetagning med positivt resultat.
Bekræftede tilfælde: 					Antal bekræftede tilfælde
Befolkningstørrelse: 					Antal af personer tilhørende den given vaccinestatus
Bekræftede tilfælde pr. 100.000: 			Antal bekræftede tilfælde divideret med den given befolkningsstørrelse ganget med 100.000.



Kolonner:
Fil: 03_bekræftede_tilfælde_pr_vacc_status_pr_aldersgrp_de_seneste_7_dage
------------------------------------------------------------------------------
Denne benyttes til at beregne alle bekræftede tilfælde opgørelser for de seneste 7 dage, herunder cirkeldiagrammet og søjlediagrammet. 

Kolonner:
Aldersgruppe:						Aldersgruppen ved vaccination
Vaccinationsstatus:					Vaccinationsstatus ved prøvetagning.
Bekræftede tilfælde: 					Antal bekræftede tilfælde
Befolkningstørrelse: 					Antal af personer tilhørende den given vaccinestatus
Bekræftede tilfælde pr. 100.000: 			Antal bekræftede tilfælde divideret med den given befolkningsstørelse ganget med 100.000.





Fil: 04_bekræftede_tilfælde_pr_vaccinationsstatus_pr_aldersgrp_pr_uge
------------------------------------------------------------------------------
Denne benyttes til at beregne alle bekræftede tilfælde opgørelser pr. uge, herunder epi-kurverne. 

Kolonner:
Uge:							En uge er 7 dage og er opgjort fra mandag til søndag.
Aldersgruppe:						Aldersgruppen ved vaccination
Vaccinationsstatus:					Vaccinationsstatus ved prøvetagning.
Bekræftede tilfælde: 					Antal bekræftede tilfælde
Befolkningstørrelse: 					Antal af personer tilhørende den given vaccinestatus
Bekræftede tilfælde pr. 100.000: 			Antal bekræftede tilfælde divideret med den given befolkningsstørelse ganget med 100.000.









Fil: 05_nyindlagte_pr_vaccinationsstatus_pr_aldersgrp_de_seneste_7_dage
------------------------------------------------------------------------------
Denne benyttes til at beregne alle indlæggelses opgørelser for de seneste 7 dage, herunder cirkeldiagrammet og søjlediagrammet. 

Kolonner:
Aldersgruppe:						Aldersgruppen ved vaccination
Vaccinationsstatus:					Vaccinationsstatus ved prøvetagning.
Antal nyindlagte: 					Antal af nyindlagte
Befolkningstørrelse: 					Antal af personer tilhørende den given vaccinestatus
Nyindlagte pr. 100.000: 				Antal nyindlagte divideret med den given befolkningsstørelse ganget med 100.000.






Fil: 06_nyindlagte_pr_vaccinationsstatus_pr_aldersgrp_pr_uge
------------------------------------------------------------------------------
Denne benyttes til at beregne alle indlæggelses opgørelser pr. uge, herunder epi-kurverne. 

Kolonner:
Uge:							En uge er 7 dage og er opgjort fra mandag til søndag.
Aldersgruppe:						Aldersgruppen ved vaccination
Vaccinationsstatus:					Vaccinationsstatus ved prøvetagning.
Antal nyindlagte: 					Antal af nyindlagte
Befolkningstørrelse: 					Antal af personer tilhørende den given vaccinestatus
Nyindlagte pr. 100.000: 				Antal nyindlagte divideret med den given befolkningsstørelse ganget med 100.000.


