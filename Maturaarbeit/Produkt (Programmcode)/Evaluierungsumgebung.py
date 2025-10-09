from Belohnungsfunktion_1 import KäsekästchenEnv

class KäsekästchenEvaluierungEnv(KäsekästchenEnv):
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
                        self.belohnung -= 1
                        self.spieler2_score +=1
                        kästchen = True

########################################## Unterschiede zu Belohnungsfunktion 1 ##########################################
        self.ist_dritter_strich_eines_kaestchens = [] # Eine Liste, die potenzielle dritte Striche eines Kästchens sammelt
        self.gegner_striche = [] # Eine Liste, die erste und zweite Striche eines Kästchens sammelt
        self.moegliche_kaestchen_zuege = [] # Eine Liste, die mögliche vierte Striche eines Kästchens sammelt
        # Hier wird gezählt, wie viele Striche ein Kästchen hat und basierend darauf werden die Koordinaten von noch
        # nicht gezogenen Strichen in die verschiedenen Listen einsortiert:
##########################################################################################################################
        
        for y in range(self.anzahl_p_h - 1):
            for x in range(self.anzahl_p_v - 1):
                striche = 0
                if self.spielfeld[x, y, 0] != 0: striche += 1
                if self.spielfeld[x, y, 1] != 0: striche += 1
                if self.spielfeld[x + 1, y, 0] != 0: striche += 1
                if self.spielfeld[x, y + 1, 1] != 0: striche += 1

########################################## Unterschiede zu Belohnungsfunktion 1 ##########################################
                if striche == 0 or striche == 1:
                    if self.spielfeld[x, y, 0] == 0 and not y == (self.anzahl_p_h - 1):
                        self.gegner_striche.append((x, y, 0))
                    if self.spielfeld[x, y, 1] == 0 and not x == (self.anzahl_p_v - 1):
                        self.gegner_striche.append((x, y, 1))
                    if self.spielfeld[x + 1, y, 0] == 0 and not y == (self.anzahl_p_h - 1) and not x == (self.anzahl_p_v - 1):
                        self.gegner_striche.append((x + 1, y, 0))
                    if self.spielfeld[x, y + 1, 1] == 0 and not y == (self.anzahl_p_h - 1) and not x == (self.anzahl_p_v - 1):
                        self.gegner_striche.append((x, y + 1, 1))
                
                if striche == 2:
                    if self.spielfeld[x, y, 0] == 0 and not y == (self.anzahl_p_h - 1):
                        self.ist_dritter_strich_eines_kaestchens.append((x, y, 0))
                    if self.spielfeld[x, y, 1] == 0 and not x == (self.anzahl_p_v - 1):
                        self.ist_dritter_strich_eines_kaestchens.append((x, y, 1))
                    if self.spielfeld[x + 1, y, 0] == 0 and not y == (self.anzahl_p_h - 1) and not x == (self.anzahl_p_v - 1):
                        self.ist_dritter_strich_eines_kaestchens.append((x + 1, y, 0))
                    if self.spielfeld[x, y + 1, 1] == 0 and not y == (self.anzahl_p_h - 1) and not x == (self.anzahl_p_v - 1):
                        self.ist_dritter_strich_eines_kaestchens.append((x, y + 1, 1))
##########################################################################################################################
                        
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
        # Ein erster oder zweiter Strich eines Kästchens, kann gleichzeitig potenziell ein dritter eines weiteren Kästchens sein
        # Aus diesem Grund wird hier eine weitere Liste von ersten und zweiten Strichen erstellt,
        # welche jedoch solche gefährlichen Striche herausfiltert und doppelte einträge aussortiert.
        self.sichere_gegner_striche = []
        for strich in self.gegner_striche:
            if strich in self.ist_dritter_strich_eines_kaestchens or strich in self.sichere_gegner_striche:
                pass
            else:
                self.sichere_gegner_striche.append(strich)
        
        # Damit möglichst verhindert werden kann, dass der Agent eine lange Kette öffnet, wird eine Liste an dritten Strichen
        # gemacht, welche jedoch nur einmal in der Liste der dritten Striche vorkommen. Somit werden Striche, die mitten
        # in einer Kette sind unterbunden. Der Beginn und das Ende von langen Ketten, können jedoch nicht herausgefiltert werden.
        # Somit wird also einfach die Wahrscheinlichkeit minimiert, dass der Gegner eine lange Kette öffnet.
        self.gegner_dritte_striche = []
        for strich in self.ist_dritter_strich_eines_kaestchens:
            if self.ist_dritter_strich_eines_kaestchens.count(strich) == 1:
                self.gegner_dritte_striche.append(strich)
##########################################################################################################################
                
        if not kästchen:
            if self.am_zug == "agent":
                self.am_zug = "spieler2"
            else:
                self.am_zug = "agent"


    def spieler2_spielzug(self):
        möglich = False
        if self.render_mode == None:
            if len(self.moegliche_kaestchen_zuege) > 0:
                strich_nr = self.np_random.integers(0, len(self.moegliche_kaestchen_zuege))
                x_s2, y_s2, z_s2 = self.moegliche_kaestchen_zuege[strich_nr]
                self.spielfeld[x_s2, y_s2, z_s2] = 1
                self.anzahl_spielzüge += 1
                
########################################## Unterschiede zu Belohnungsfunktion 1 ##########################################
            # 2. Priorität: Erste und Zweite Striche von Kästchen.
            elif len(self.sichere_gegner_striche) > 0:
                strich_nr = self.np_random.integers(0, len(self.sichere_gegner_striche))
                x_s2, y_s2, z_s2 = self.sichere_gegner_striche[strich_nr]
                self.spielfeld[x_s2, y_s2, z_s2] = 1
                self.anzahl_spielzüge += 1
                
            # 3. Priorität: Dritter Strich eines Kästchens, der nicht mitten in einer Kette liegt.
            elif len(self.gegner_dritte_striche) > 0:
                strich_nr = self.np_random.integers(0, len(self.gegner_dritte_striche))
                x_s2, y_s2, z_s2 = self.gegner_dritte_striche[strich_nr]
                self.spielfeld[x_s2, y_s2, z_s2] = 1
                self.anzahl_spielzüge += 1
                
            # 4. Priorität: Ein dritter Strich eines Kästchens.
            else:
                strich_nr = self.np_random.integers(0, len(self.ist_dritter_strich_eines_kaestchens))
                x_s2, y_s2, z_s2 = self.ist_dritter_strich_eines_kaestchens[strich_nr]
                self.spielfeld[x_s2, y_s2, z_s2] = 1
                self.anzahl_spielzüge += 1
##########################################################################################################################
        
        elif self.render_mode == "human":
            while not möglich:
                 for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        maus_x, maus_y = pygame.mouse.get_pos()                
                        klick_toleranz = 15
                        for y in range(self.anzahl_p_h):
                            for x in range(self.anzahl_p_v):
                                punkt_x = (y + 1) * self.pixel_pro_einheit_h
                                punkt_y = (x + 1) * self.pixel_pro_einheit_v
                                if y < self.anzahl_p_h - 1:
                                    x_im_bereich = punkt_x < maus_x < (punkt_x + self.pixel_pro_einheit_h)
                                    y_im_bereich = (punkt_y - klick_toleranz) < maus_y < (punkt_y + klick_toleranz)
                                    if x_im_bereich and y_im_bereich:
                                        if self.spielfeld[x, y, 0] == 0:
                                            self.spielfeld[x, y, 0] = 1
                                            self.anzahl_spielzüge += 1
                                            möglich = True
                                            break
                                if x < self.anzahl_p_v - 1:
                                    y_im_bereich = punkt_y < maus_y < (punkt_y + self.pixel_pro_einheit_v)
                                    x_im_bereich = (punkt_x - klick_toleranz) < maus_x < (punkt_x + klick_toleranz)
                                    if y_im_bereich and x_im_bereich:
                                        if self.spielfeld[x, y, 1] == 0:
                                            self.spielfeld[x, y, 1] = 1
                                            self.anzahl_spielzüge += 1
                                            möglich = True
                                            break
                            if möglich:
                                break
        if self.anzahl_spielzüge == self.max_spielzüge:
            self.fertig = True