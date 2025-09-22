# Code basierend auf https://github.com/Farama-Foundation/gymnasium-env-template
# Copyright (c) 2023 Farama Foundation
# Lizenz: MIT License (https://opensource.org/licenses/MIT)
# Änderungen vorgenommen von Fabian Rütschi, 2025

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Discrete, MultiDiscrete
import pygame
import numpy as np
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3 import DQN

class KäsekästchenEnv(gym.Env):
    
    
########################################## Folgender Codeabschnitt wurde übernommen ##########################################
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
##############################################################################################################################            


    def __init__(self, anzahl_punkte_horizontal, anzahl_punkte_vertikal, render_mode=None):
        self.anzahl_p_h = anzahl_punkte_horizontal # Grösse des Spielfelds.
        self.anzahl_p_v = anzahl_punkte_vertikal # Grösse des Spielfelds.

        # 3D-Array, das als Beobachtung an Agent weitergegeben wird. Enthält Werte zwischen 0 und 3.
        self.observation_space = spaces.Box(low=0, high=3, shape=(self.anzahl_p_v, self.anzahl_p_h, 2), dtype=np.uint8)

        # Der action_space wid als die Anzahl der möglichen Aktionen, also die Anzahl Striche, definiert.
        self.action_space = Discrete((self.anzahl_p_h - 1) * self.anzahl_p_v + (self.anzahl_p_v - 1) * self.anzahl_p_h)
        
        # Damit das Spielfeld erstellt wird und alle Punkte den Wert 0 erhalten, wird die reset() Funktion ausgeführt.        
        self.reset()
        
        
########################################## Folgender Codeabschnitt wurde übernommen ##########################################
        # Überprüfen und festlegen des Render-Modus.
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Diese Eigenschaften werden nur für den "human" Render-Modus benutzt. Deshalb sind sie standardmässig None.      
        self.window = None
        self.clock = None
##############################################################################################################################            


    def _get_obs(self):
        return self.spielfeld # Gibt das 3D-Array des aktuellen Spielfelds zurück.
    
    def _get_info(self):
        # Überprüft, ob beim letzten Zug ein neues Kästchen geschlossen wurde.
        # Ist dies der Fall, wird je nach Spieler der Agenten-Reward erhöht oder gesenkt.
        # Wurde kein Kästchen geschlossen, wechselt der Spieler, der am Zug ist.
        kästchen = False
        for y_koordinate in range(self.anzahl_p_h - 1):
            for x_koordinate in range(self.anzahl_p_v - 1):
                # Haben bei einem Kästchen der linke und obere Strich den Wert 1
                # und der rechte und untere Strich sind nicht 0, wurde durch die Aktion ein neues Kästchen geschlossen.
                if (
                    self.spielfeld[x_koordinate, y_koordinate, 0] == 1 and
                    self.spielfeld[x_koordinate, y_koordinate, 1] == 1 and
                    self.spielfeld[x_koordinate + 1, y_koordinate, 0] != 0 and
                    self.spielfeld[x_koordinate, y_koordinate + 1, 1] != 0
                ):
                     # Ist der aktuelle Spieler der Agent, wird der Wert des oberen und linken Strichs auf 2 gesetzt. Ist der
                     # aktuelle Spieler nicht der Agent -> 3, somit kann das Kästchen in Zukunft nicht mehr als neu erkannt werden.
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

        self.moegliche_kaestchen_zuege = [] # Eine Liste, die mögliche vierte Striche eines Kästchens sammelt
        
        # Hier wird gezählt, wie viele Striche ein Kästchen hat und basierend darauf werden die Koordinaten von noch
        # nicht gezogenen Strichen in eine Liste einsortiert:
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
                
        # Wurde kein neues Kästchen gefunden, wechselt der Spieler.
        if not kästchen:
            if self.am_zug == "agent":
                self.am_zug = "spieler2"
            else:
                self.am_zug = "agent"

    def reset(self, seed=None, options=None):
        # Damit das Verhalten des Spieler 2 reproduzierbar ist, kann seed=(eine Zahl) gesetzt werden,
        # somit würde der Spieler 2 bei jedem Spiel gleiche Positionen in gleicher Reihenfolge wählen.
        super().reset(seed=seed)

        self.spielfeld = np.zeros(shape=(self.anzahl_p_v, self.anzahl_p_h, 2), dtype=np.uint8) # Alle Punkte werden auf 0 gesetzt.
        
        self.anzahl_spielzüge = 0 # Anzahl_Spielzüge wird auf 0 gesetzt
        
        self.am_zug = "agent" # Da der Agent an step() immer auch schon seine action übergibt, muss immer der Agent beginnnen.
        self.fertig = False
        self.agent_score = 0
        self.spieler2_score = 0
        
        # Berechnung der maximal möglichen Anzahl Spielzüge.
        self.max_spielzüge = (self.anzahl_p_h - 1) * self.anzahl_p_v + (self.anzahl_p_v - 1) * self.anzahl_p_h
        
        observation = self._get_obs() 

        if self.render_mode == "human":
            # Für die Darstellung wird jeweils errechnet, wie viele Pixel eine Einheit hat. Es gibt pro Achse die Anzahl Punkte + 2 Einheiten.
            self.pixel_pro_einheit_h = (self.anzahl_p_h * 100)/(self.anzahl_p_h + 2)
            self.pixel_pro_einheit_v = (self.anzahl_p_v * 100)/(self.anzahl_p_v + 2)
            self.window_size_x = self.anzahl_p_h * 100
            self.window_size_y = self.anzahl_p_v * 100
            self._render_frame()
        
        info = {}
        return observation, info
    
    def aktion_umrechnen(self, aktion):
        # Da der Agent nur eine Zahl als Output liefert, müssen daraus noch die Koordinaten für das Array berechnet werden.
        if aktion < self.anzahl_p_v * (self.anzahl_p_h - 1):
            z = 0
            x = aktion % self.anzahl_p_v
            y = aktion // self.anzahl_p_v
        else:
            z = 1
            aktions_diff = aktion - self.anzahl_p_v * (self.anzahl_p_h - 1)
            x = aktions_diff % (self.anzahl_p_v - 1)
            y = aktions_diff // (self.anzahl_p_v - 1)
        return [x, y, z]
    
    def agenten_spielzug(self, aktion):
        aktion = self.aktion_umrechnen(aktion)
        x, y, z = aktion # Den Koordinaten des ausgewählten Punktes werden Variablen gegeben.
        
        if self.spielfeld[x, y, z] != 0: # Überprüfung, ob Punkt schon belegt ist.
            self.belohnung -= 2
            # Spielt man selbst gegen den Agenten, wird bei ungültigen Zügen des Agenten ein zufälliger Zug gewählt.
            if self.render_mode == "human":
                möglich = False
                while not möglich:
                    aktion = self.np_random.integers(self.max_spielzüge)
                    x, y, z = self.aktion_umrechnen(aktion)
                    if self.spielfeld[x, y, z] != 0: # Überprüfung, ob Punkt schon belegt ist.
                        pass
                    else:
                        self.spielfeld[x, y, z] = 1 # Der Wert des zufällig gewählten Punktes wird auf 1 gesetzt.
                        self.anzahl_spielzüge += 1
                        möglich = True
        else:
            self.spielfeld[x, y, z] = 1 # Der Wert des vom Agenten gewählten Punktes wird auf 1 gesetzt.
            self.anzahl_spielzüge += 1
        
        if self.anzahl_spielzüge == self.max_spielzüge: # Sobald die maximale Anzahl an Spielzügen erreicht wurde, ist ein Spiel zu Ende.
            self.fertig = True
        
    def spieler2_spielzug(self):
        # Es werden solange zufällige Koordinaten gewählt, bis ein noch unbesetzter Strich gewählt wird.
        # Damit diese reproduzierbar sind, wird self.np_random.integers aufgerufen und der in reset() definierte Seed verwendet.
        möglich = False
        if self.render_mode == None:
            # 1. Priorität: Kästchen schliessen.
            if len(self.moegliche_kaestchen_zuege) > 0:
                strich_nr = self.np_random.integers(0, len(self.moegliche_kaestchen_zuege))
                x_s2, y_s2, z_s2 = self.moegliche_kaestchen_zuege[strich_nr]
                self.spielfeld[x_s2, y_s2, z_s2] = 1
                self.anzahl_spielzüge += 1
                möglich = True
                
            # 2. Priorität: Es wird zufällig ein Strich gewählt
            while not möglich:
                x_s2 = self.np_random.integers(0, self.anzahl_p_v)
                y_s2 = self.np_random.integers(0, self.anzahl_p_h)
                z_s2 = self.np_random.integers(0, 2)
                if self.spielfeld[x_s2, y_s2, z_s2] != 0: # Überprüfung, ob Punkt schon belegt ist.
                    pass
                elif z_s2 == 0 and y_s2 == (self.anzahl_p_h - 1): # Überprüfung auf Zulässigkeit.
                    pass
                elif z_s2 == 1 and x_s2 == (self.anzahl_p_v - 1): # Überprüfung auf Zulässigkeit.
                    pass
                else:
                    self.spielfeld[x_s2, y_s2, z_s2] = 1 # Der Wert des zufällig gewählten Punktes wird auf 1 gesetzt.
                    self.anzahl_spielzüge += 1
                    möglich = True
        
        
        # Im "human"-Modus wird die Aktion des Gegners durch das Anklicken eines nicht-vorhandenen Strichs mit der Maus getätigt.
        elif self.render_mode == "human":
            while not möglich:
                 for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        exit()
                    
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        maus_x, maus_y = pygame.mouse.get_pos()                
                        klick_toleranz = 15
                    
                        # Es wird errechnet, welchen Koordinaten die Mausposition entspricht.
                        for y in range(self.anzahl_p_h):
                            for x in range(self.anzahl_p_v):
                                punkt_x = (y + 1) * self.pixel_pro_einheit_h
                                punkt_y = (x + 1) * self.pixel_pro_einheit_v
                                
                                # Es wird geprüft, ob der Mauszeiger im Toleranzbereich eines horizontalen Strichs ist.
                                if y < self.anzahl_p_h - 1:
                                    x_im_bereich = punkt_x < maus_x < (punkt_x + self.pixel_pro_einheit_h)
                                    y_im_bereich = (punkt_y - klick_toleranz) < maus_y < (punkt_y + klick_toleranz)
                                    
                                    # Die Eingabe wird getätigt.
                                    if x_im_bereich and y_im_bereich:
                                        if self.spielfeld[x, y, 0] == 0:
                                            self.spielfeld[x, y, 0] = 1
                                            self.anzahl_spielzüge += 1
                                            möglich = True
                                            break

                                # Es wird geprüft, ob der Mauszeiger im Toleranzbereich eines vertikalen Strichs ist.
                                if x < self.anzahl_p_v - 1:
                                    y_im_bereich = punkt_y < maus_y < (punkt_y + self.pixel_pro_einheit_v)
                                    x_im_bereich = (punkt_x - klick_toleranz) < maus_x < (punkt_x + klick_toleranz)
                                    
                                    # Die Eingabe wird getätigt.
                                    if y_im_bereich and x_im_bereich:
                                        if self.spielfeld[x, y, 1] == 0:
                                            self.spielfeld[x, y, 1] = 1
                                            self.anzahl_spielzüge += 1
                                            möglich = True
                                            break
                            
                            if möglich:
                                break
                

        if self.anzahl_spielzüge == self.max_spielzüge: # Sobald die maximale Anzahl an Spielzügen erreicht wurde, ist ein Spiel zu Ende.
            self.fertig = True

    def step(self, action):
        # Die Belohnung wird zu Beginn jedes Zeitschritts auf 0 gesetzt. 
        self.belohnung = 0


        if self.am_zug == "agent" and not self.fertig: # Agent macht immer nur ein Zug aufs Mal, da er der Step Funktion eine neue Aktion geben muss.
            self.agenten_spielzug(action)
            self._get_info() # Es wird überprüft, welche Änderungen der Zug hervorgerufen hat.
            if self.render_mode == "human":
                self._render_frame()
            
        while self.am_zug == "spieler2" and not self.fertig: # Wenn der Gegner spielt, darf er so lange Punkte setzen, bis er nicht mehr am Zug ist.
            self.spieler2_spielzug()
            self._get_info() # Es wird überprüft, welche Änderungen der Zug hervorgerufen hat.
            if self.render_mode == "human":
                self._render_frame()
                            
        observation = self._get_obs() # Der Zustand des 3D Arrays wird als Observation dem Agenten übergeben.
        
        terminated = self.fertig # Gibt die Information weiter, ob das Spiel beendet ist.
        
        info = {} # Stable_Baselines3 erwartet eine info, deshalb muss diese zurückgegeben werden.
        
        if self.fertig and self.agent_score > self.spieler2_score:
            self.belohnung += 1 # Belohnung für einen Sieg
            info["is_success"] = True # Die Information, ob der Agent gewonnen hat wird weitergegeben.
        elif self.fertig and self.agent_score < self.spieler2_score:
            self.belohnung -= 1 # Strafe für eine Niederlage
            info["is_success"] = False # Die Information, ob der Agent verloren hat wird weitergegeben.
            
        reward = self.belohnung # Die Variable self.belohnung übergibt ihren Wert an reward, der dann an den Agenten gegeben wird.
        
        return observation, reward, terminated, False, info
    
    
########################################## Folgender Codeabschnitt wurde übernommen ##########################################
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size_x, self.window_size_y))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
##############################################################################################################################            


        canvas = pygame.Surface((self.anzahl_p_h * 100, self.anzahl_p_v * 100))
        canvas.fill((255, 255, 255))
        
        # Zuerst werden die geschlossenen Kästchen eingefärbt:
        for y in range(self.anzahl_p_h - 1):
            for x in range(self.anzahl_p_v - 1):
                # Es wird geprüft, ob das Kästchen geschlossen ist. Wenn der Punkt auf x-Ebene 1 und 2 den Wert 2 hat, gehört das Kästchen dem Agent, wenn es den Wert 3 hat, gehört es dem Spieler2.
                spieler_nummer = self.spielfeld[x, y, 0]
                if spieler_nummer == 2: # Agent
                    farbe = (0, 0, 255)
                elif spieler_nummer == 3: # Spieler 2
                    farbe = (255, 0, 0)
                else:
                    continue # Das Kästchen ist noch nicht geschlossen, also wird auch nichts gezeichnet. Es wird automatisch der nächste Wert für z respektive y genommen, ohne den unteren Code mit dem Zeichnen der Kästchen auszuführen.
                
                # Zeichnen der Kästchen:
                käst_x = (y + 1) * self.pixel_pro_einheit_h
                käst_y = (x + 1) * self.pixel_pro_einheit_v
                käst_breite = self.pixel_pro_einheit_h
                käst_höhe = self.pixel_pro_einheit_v
                pygame.draw.rect(canvas, farbe, pygame.Rect(käst_x, käst_y, käst_breite, käst_höhe))
        
        # Dann werden die Punkte gezeichnet:
        for y in range(1, self.anzahl_p_h + 1):
            for x in range(1, self.anzahl_p_v + 1):
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    (self.pixel_pro_einheit_h * y, self.pixel_pro_einheit_v * x),
                    10
                )

        # Als Nächstes werden die horizontalen Linien zwischen den Punkten geichnet:
        for y in range(self.anzahl_p_h - 1):
            for x in range(self.anzahl_p_v):
                if self.spielfeld[x, y, 0] != 0:
                    punkt1 = ((y + 1) * self.pixel_pro_einheit_h, (x + 1) * self.pixel_pro_einheit_v)
                    punkt2 = ((y + 2) * self.pixel_pro_einheit_h, (x + 1) * self.pixel_pro_einheit_v)
                    pygame.draw.line(canvas, (0, 0, 0), punkt1, punkt2, 5)
                    
        # Dann werden die vertikalen Linien gezeichnet:
        for y in range(self.anzahl_p_h):
            for x in range(self.anzahl_p_v - 1):
                if self.spielfeld[x, y, 1] != 0:
                    punkt1 = ((y + 1) * self.pixel_pro_einheit_h, (x + 1) * self.pixel_pro_einheit_v)
                    punkt2 = ((y + 1) * self.pixel_pro_einheit_h, (x + 2) * self.pixel_pro_einheit_v)
                    pygame.draw.line(canvas, (0, 0, 0), punkt1, punkt2, 5)
        
       
########################################## Folgender Codeabschnitt wurde übernommen ##########################################
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()         
##############################################################################################################################            


