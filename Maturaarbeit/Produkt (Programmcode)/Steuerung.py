# Einzelne Codezeilen übernommen aus der offiziellen Stable-Baselines3 Dokumentation:
# https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
# https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
# https://stable-baselines3.readthedocs.io/en/master/common/monitor.html#module-stable_baselines3.common.monitor
# Lizenz: MIT License (https://opensource.org/licenses/MIT)

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
import time


seed = 0 # Durch einen festen Seedwert wird die Reproduzierbarkeit sichergestellt.


# Rückrufs-Funktion, damit die Gewinnrate berechnet wird.
def gewinnrate_evaluierung(modell, umgebung, anzahl_spiele):
    global seed
    siege = 0
    spielbelohnung_liste = []
    
    for _ in range(anzahl_spiele):
        obs, info = umgebung.reset(seed=seed+_)
        fertig = False
        spielbelohnung = 0
        
        while not fertig:
            action, _ = modell.predict(obs, deterministic=True)
            obs, reward, fertig, truncated, info = umgebung.step(action)
            spielbelohnung += reward

            if fertig:
                spielbelohnung_liste.append(spielbelohnung)
                if info['is_success']:
                  siege += 1

        gewinnrate = siege / anzahl_spiele
    
    return spielbelohnung_liste, gewinnrate


# Entsprechende Umgebung wird importiert.
def umgebung_importieren(belohnungsfunktion):
    if "B2" in belohnungsfunktion:
        from Belohnungsfunktion_2 import KäsekästchenEnv
    elif "B3" in belohnungsfunktion:
        from Belohnungsfunktion_3 import KäsekästchenEnv
    elif "B1" in belohnungsfunktion:
        from Belohnungsfunktion_1 import KäsekästchenEnv
    else:
        from Belohnungsfunktion_1 import KäsekästchenEnv
        print("Als Umgebung wurde automatisch B1 gewählt.")
    
    # Evaluierungsumgebung wird importiert.
    from Evaluierungsumgebung import KäsekästchenEvaluierungEnv
    
    return KäsekästchenEnv, KäsekästchenEvaluierungEnv


# Funktion, um Nutzer nach Modus zu fragen.
def user_abfrage():
    modus = input("""
Geben Sie eine Zahl ein, um eine Aktion zu starten:
1: Neues Modell trainieren.
2: Vorhandendes Modell weitertrainieren.
3: Modell evaluieren lassen.
4: Selber Gegen ein trainiertes Modell spielen.
Modus: """)
    if "1" in modus:
        modus_1()
    elif "2" in modus:
        modus_2()
    elif "3" in modus:
        modus_3()
    elif "4" in modus:
        modus_4()
    else:
        print("Unzulässige Eingabe")
        user_abfrage()


# Mit Modus 1 können neue Modelle trainiert werden. 
def modus_1():
    global seed
    
    belohnungsfunktion = input("Wählen Sie mit welcher Belohnungsfunktion (B1, B2, B3) Sie trainieren wollen: ")
    
    #Umgebungen werden importiert
    KäsekästchenEnv, KäsekästchenEvaluierungEnv = umgebung_importieren(belohnungsfunktion)
        
    umgebung = KäsekästchenEnv(4, 4) # Trainingsumgebung wird initialisiert.
    obs, info = umgebung.reset(seed=seed)
    
    evaluierungs_umgebung = KäsekästchenEvaluierungEnv(4, 4) # Evaluierungsumgebung wir initialisiert.
    evaluierungs_umgebung = Monitor(evaluierungs_umgebung, info_keywords=("is_success",))
    obs, info = evaluierungs_umgebung.reset(seed=seed)    
    
    parameter = input("Wenn Sie die Hyperparameter anpassen wollen, geben Sie bitte 1 ein, sonst 0: ")
    if "1" in parameter:
        try:
            lernrate = float(input("Definieren Sie die Lernrate (Standard: 0.0001, Range: 0.00005 - 0.001): "))
            gamma = float(input("Definieren Sie den Diskontierungsfaktor (Standard: 0.99, Range: 0.8 - 0.999): "))
            explorationsphasenanteil = float(input("Definieren Sie den Explorationsphasenanteil (Standard: 0.1, Range: 0.05 - 0.25): "))
            zeitschritte = int(input("Wie viele Zeitschritte wollen Sie trainieren? (Standard: 2500000, Range: 200000 - 3000000): "))
        except ValueError:
            print("Unzulässige Eingabe")
            return modus_1()
    else:
        # Standardwerte:
        lernrate = 0.0001
        gamma = 0.99
        explorationsphasenanteil = 0.1
        zeitschritte = 2500000
        
    modell = DQN(
        'MlpPolicy',
        umgebung,
        learning_rate=lernrate,
        gamma=gamma,
        exploration_fraction=explorationsphasenanteil,
        buffer_size=100000,                      
        verbose=1,                               
        tensorboard_log="./Käsekästchen_Models/" 
    )
    
    # Dateinamen mit den Parametern.
    dateiname = f"{belohnungsfunktion}_lernrate{lernrate}_gamma{gamma}_explorationsphasenanteil{explorationsphasenanteil}_zeitschritt{zeitschritte}"
    dateiname = dateiname.replace('.', ',')
    
    # Parameter für die zyklische Evaluierung werden festgelegt.
    evaluierungs_callback = EvalCallback(evaluierungs_umgebung, best_model_save_path=f"./logs/{dateiname}",
                                         log_path="./logs/", eval_freq=10000, n_eval_episodes=100,
                                         deterministic=True, render=False)
    
    # Trainings-Metriken-Logger wird konfiguriert.
    tmp_path = f"./logs/tmp/{dateiname}/"
    logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    
    modell.set_logger(logger) # Logger wird gesetzt.
    
    input("""
Um Statistiken zum Training zu sehen, öffnen Sie den Pfad dieser Datei im Windows-Explorer, tippen Sie in die Suchleiste:
cmd
Danach öffnet sich das Terminal, tippen Sie dann Folgendes ein:
tensorboard --logdir ./logs/tmp/ --reload_interval 30
Drücken Sie die Enter Taste als Eingabe auf diese Nachricht, um das Training zu starten: 
""")
    
    modell.learn(total_timesteps=zeitschritte, callback=evaluierungs_callback, log_interval=4) # Das Modell wird trainiert.
    modell.save(dateiname) # Das fertige Modell wird abgespeichert.
    print(dateiname)
    
    time.sleep(5)
    user_abfrage()
    
    
# Mit Modus 2 können bereits trainierte Modelle noch mehr trainiert werden.    
def modus_2():
    global seed
    
    modell_laden = input("Geben Sie den Dateinamen des Modells ein, das Sie laden wollen: ")
    
    #Umgebungen werden importiert
    KäsekästchenEnv, KäsekästchenEvaluierungEnv = umgebung_importieren(modell_laden)
    
    umgebung = KäsekästchenEnv(4, 4)
    obs, info = umgebung.reset(seed=seed)
    
    evaluierungs_umgebung = KäsekästchenEvaluierungEnv(4, 4) # Evaluierungsumgebung wir initialisiert.
    evaluierungs_umgebung = Monitor(evaluierungs_umgebung, info_keywords=("is_success",))
    obs, info = evaluierungs_umgebung.reset(seed=seed)
    
    try:
        zeitschritte = int(input("Wie viele Zeitschritte wollen Sie trainieren? "))
    except ValueError:
        print("Unzulässige Eingabe")
        return modus_2()
    
    # Dateinamen mit den Parametern.
    dateiname = f"V2_{modell_laden}_V2_zeitschritt{zeitschritte}"
    dateiname = dateiname.replace('.', ',')
    
    # Parameter für die zyklische Evaluierung werden festgelegt.
    evaluierungs_callback = EvalCallback(evaluierungs_umgebung, best_model_save_path=f"./logs/{dateiname}",
                                         log_path="./logs/", eval_freq=10000, n_eval_episodes=100,
                                         deterministic=True, render=False)
    
    # Trainings-Metriken-Logger wird konfiguriert.
    tmp_path = f"./logs/tmp/{dateiname}/"
    logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    
    modell = DQN.load(modell_laden) # Modell wird geladen.
    modell.set_env(umgebung) # Umgebung wird gesetzt.
    modell.set_logger(logger) # Logger wird gesetzt.
    
    input("""
Um Statistiken zum Training zu sehen, öffnen Sie den Pfad dieser Datei im Windows-Explorer, tippen Sie in die Suchleiste:
cmd
Danach öffnet sich das Terminal, tippen Sie dann Folgendes ein:
tensorboard --logdir ./logs/tmp/ --reload_interval 30
Drücken Sie die Enter Taste als Eingabe auf diese Nachricht, um das Training zu starten: 
""")
    
    modell.learn( # Modell beginnt zu lernen.
        total_timesteps=zeitschritte,
        callback=evaluierungs_callback,
        log_interval=10,
        reset_num_timesteps=False
    )
    
    modell.save(dateiname) # Das fertige Modell wird abgespeichert.
    print(dateiname)
    
    time.sleep(5)
    user_abfrage()


# Mit Modus 3 können trainierte Modelle in der Evaluierungsumgebung final evaluiert werden.
def modus_3():
    global seed
    
    try:
        modell_laden = input("Geben Sie den Dateinamen des Modells ein, das Sie laden wollen: ")
        modell = DQN.load(modell_laden)
        anzahl_spiele = int(input("Über wie viele Spiele wollen Sie evaluieren? (Standard: 1000): "))
    except ValueError:
        print("Unzulässige Eingabe")
        return modus_3()

    # Evaluierungsumgebung wird importiert.
    from Evaluierungsumgebung import KäsekästchenEvaluierungEnv
    umgebung = KäsekästchenEvaluierungEnv(4, 4) # Evaluierungsumgebung wird initialisiert.
    obs, info = umgebung.reset(seed=seed)
    umgebung = Monitor(umgebung, info_keywords=("is_success",)) # Trackt die Anzahl der Gewinne bei der Evaluierung für gewinnrate_evaluierung()
    modell.set_env(umgebung)
    
    belohnung_durchschnitt, belohnung_standardabweichung = evaluate_policy(
        modell, 
        umgebung, 
        n_eval_episodes=anzahl_spiele, 
        deterministic=True
    )
    
    # Die mit evaluate_policy die Gewinnrate nicht getrackt werden kann, wurde dazu eine extra Fuktion programmiert.
    # Diese Funktion evaluiert jedoch die Gewinnrate separat von evaluate_policy
    kumulierte_belohnungen, gewinnrate = gewinnrate_evaluierung(modell, umgebung, anzahl_spiele)
    
    print(f"""
Liste der kumulierten Belohnungen jedes Spiels:

{kumulierte_belohnungen}

Evaluierungswerte des Modell {modell_laden}:
Durchschnittliche Belohnung:  {belohnung_durchschnitt}
Standardabweichung Belohnung: {belohnung_standardabweichung}
Gewinnrate des Agenten: {gewinnrate}
""")
    
    time.sleep(3)
    user_abfrage()
        

# Mit Modus 3 kann man selbst gegen ein trainiertes Modell spielen.
def modus_4():
    try:
        modell_laden = input("Geben Sie den Dateinamen des Modells ein, das Sie laden wollen: ")
        #Umgebung wird importiert
        KäsekästchenEnv, KäsekästchenEvaluierungEnv = umgebung_importieren(modell_laden)
        zeitschritte = int(input("Wie viele Zeitschritte wollen Sie spielen? "))
    except ValueError:
        print("Unzulässige Eingabe")
        return modus_4()
    
    umgebung = KäsekästchenEnv(4, 4, render_mode="human")
    obs, info = umgebung.reset()
    
    modell = DQN.load(modell_laden)
    for _ in range(zeitschritte):
        action, _ = modell.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = umgebung.step(action)
        umgebung.render() 
        time.sleep(0.5)
        if terminated or truncated:
            print("Das Spiel ist fertig. Agent-Score:", umgebung.agent_score, "Ihr Score:", umgebung.spieler2_score)
            time.sleep(3)
            obs, info = umgebung.reset()
    umgebung.close()
    
    user_abfrage()


# Programm wird gestartet
user_abfrage()