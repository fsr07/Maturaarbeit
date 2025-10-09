from Belohnungsfunktion_1 import KäsekästchenEnv as KäsekästchenBasisEnv

class KäsekästchenEnv(KäsekästchenBasisEnv):
    def _get_info(self):
        kästchen = False
        for y_koordinate in range(self.anzahl_p_h - 1):
            for x_koordinate in range(self.anzahl_p_v - 1):
                if (
                    self.spielfeld[x_koordinate, y_koordinate, 0] == 1 and
                    self.spielfeld[x_koordinate, y_koordinate, 1] == 1 and
                    self.spielfeld[x_koordinate + 1, y_koordinate, 0] != 0 and
                    self.spielfeld[x_koordinate, y_koordinate + 1, 1] != 0
                ):
                    if self.am_zug == "agent":
                        self.spielfeld[x_koordinate, y_koordinate, 0] = 2
                        self.spielfeld[x_koordinate, y_koordinate, 1] = 2
                        self.belohnung += 1
                        self.agent_score += 1
                        kästchen = True
                    else:
                        self.spielfeld[x_koordinate, y_koordinate, 0] = 3
                        self.spielfeld[x_koordinate, y_koordinate, 1] = 3
            
########################################## Unterschiede zu Belohnungsfunktion 1 ##########################################
                        self.belohnung -= 0.5
##########################################################################################################################
                        
                        self.spieler2_score +=1
                        kästchen = True

        self.moegliche_kaestchen_zuege = []
        for y in range(self.anzahl_p_h - 1):
            for x in range(self.anzahl_p_v - 1):
                striche = 0
                if self.spielfeld[x, y, 0] != 0: striche += 1
                if self.spielfeld[x, y, 1] != 0: striche += 1
                if self.spielfeld[x + 1, y, 0] != 0: striche += 1
                if self.spielfeld[x, y + 1, 1] != 0: striche += 1
                if striche == 3:
                    if self.spielfeld[x, y, 0] == 0:
                        self.moegliche_kaestchen_zuege.append((x, y, 0))
                    elif self.spielfeld[x, y, 1] == 0:
                        self.moegliche_kaestchen_zuege.append((x, y, 1))
                    elif self.spielfeld[x + 1, y, 0] == 0:
                        self.moegliche_kaestchen_zuege.append((x + 1, y, 0))
                    elif self.spielfeld[x, y + 1, 1] == 0:
                        self.moegliche_kaestchen_zuege.append((x, y + 1, 1))
        
########################################## Unterschiede zu Belohnungsfunktion 1 ##########################################
        # Die Anzahl der möglichen Handlungen, die der Gegner wählen kann,
        # um Kästchen zu schliessen, wird von der Belohnung * 0.5 subtrahiert.
        if self.am_zug == "agent" and not kästchen:
            self.belohnung -= 0.5 * len(self.moegliche_kaestchen_zuege)
##########################################################################################################################
        
        if not kästchen:
            if self.am_zug == "agent":
                self.am_zug = "spieler2"
            else:
                self.am_zug = "agent"